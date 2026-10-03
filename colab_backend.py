import os
import io
import time
import torch
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize FastAPI App
app = FastAPI(
    title="BharatGenFinance Colab Backend API",
    description="FastAPI Backend for FinanceParam LLM + RAG Vector Store running on Google Colab GPU",
    version="1.0.0"
)

# Enable CORS for cross-origin requests from Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration & Paths
DRIVE_BASE_DIR = "/content/drive/MyDrive/BharatGenFinance"
LOCAL_BASE_DIR = os.getcwd()

# Determine active base directory (Google Drive if mounted, else local)
if os.path.exists("/content/drive/MyDrive"):
    BASE_DIR = DRIVE_BASE_DIR
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"📁 Using Google Drive storage: {BASE_DIR}")
else:
    BASE_DIR = LOCAL_BASE_DIR
    print(f"📁 Google Drive not detected. Using local directory: {BASE_DIR}")

MODEL_STORAGE_PATH = os.path.join(BASE_DIR, "FinanceParam")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_index")
TEMP_UPLOADS_PATH = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_UPLOADS_PATH, exist_ok=True)

# Global State Variables
tokenizer = None
model = None
vector_store = None
embeddings_model = None

# Model Name on HuggingFace Hub
MODEL_NAME = "bharatgenai/FinanceParam"

# Global monkey-patch for DynamicCache to support legacy tuple indexing (past_key_values[idx])
try:
    import transformers.cache_utils
    if hasattr(transformers.cache_utils, "DynamicCache"):
        DC = transformers.cache_utils.DynamicCache
        def _dc_getitem(self, idx):
            if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
                if isinstance(idx, int) and 0 <= idx < len(self.key_cache):
                    return (self.key_cache[idx], self.value_cache[idx])
            return None
        DC.__getitem__ = _dc_getitem
        
        def _dc_len(self):
            return len(self.key_cache) if hasattr(self, "key_cache") else 0
        DC.__len__ = _dc_len
except Exception as patch_err:
    print(f"DynamicCache monkey-patch warning: {patch_err}")

def patch_remote_code_if_needed():
    # Runtime patch for HF cached file modeling_parambharatgen.py for 100% transformers 4.45+ compatibility and fast KV caching
    import glob
    search_paths = [
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/**/*.py"),
        os.path.join(BASE_DIR, "**/*.py"),
        os.path.join(LOCAL_BASE_DIR, "**/*.py")
    ]
    for spath in search_paths:
        for fpath in glob.glob(spath, recursive=True):
            if "modeling_parambharatgen.py" in fpath:
                try:
                    with open(fpath, "r") as f:
                        code = f.read()
                    patched = False
                    if 'scaling_type = self.config.rope_scaling["type"]' in code:
                        code = code.replace(
                            'scaling_type = self.config.rope_scaling["type"]',
                            'rope_sc = self.config.rope_scaling if isinstance(self.config.rope_scaling, dict) else {}; scaling_type = rope_sc.get("type", rope_sc.get("rope_type", "default"))'
                        )
                        patched = True
                    if 'scaling_factor = self.config.rope_scaling["factor"]' in code:
                        code = code.replace(
                            'scaling_factor = self.config.rope_scaling["factor"]',
                            'scaling_factor = (self.config.rope_scaling if isinstance(self.config.rope_scaling, dict) else {}).get("factor", 1.0)'
                        )
                        patched = True
                    if 'raise ValueError(f"Unknown RoPE scaling type {scaling_type}")' in code:
                        code = code.replace(
                            'raise ValueError(f"Unknown RoPE scaling type {scaling_type}")',
                            'self.rotary_emb = ParamBharatGenRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)'
                        )
                        patched = True
                    if 'if past_key_values is not None:' in code and 'to_legacy_cache' not in code:
                        code = code.replace(
                            'if past_key_values is not None:',
                            'if past_key_values is not None:\n            if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) == 0:\n                past_key_values = None\n            elif hasattr(past_key_values, "to_legacy_cache"):\n                past_key_values = past_key_values.to_legacy_cache()\n            elif hasattr(past_key_values, "key_cache"):\n                past_key_values = tuple(zip(past_key_values.key_cache, past_key_values.value_cache))\n        if past_key_values is not None and hasattr(past_key_values, "__len__") and len(past_key_values) > 0 and past_key_values[0] is not None:'
                        )
                        patched = True
                    if 'past_key_values[idx] if past_key_values is not None else None' in code:
                        code = code.replace(
                            'past_key_values[idx] if past_key_values is not None else None',
                            'past_key_values[idx] if (past_key_values is not None and hasattr(past_key_values, "__getitem__") and past_key_values[idx] is not None) else None'
                        )
                        patched = True
                    if patched:
                        with open(fpath, "w") as f:
                            f.write(code)
                        print("🔧 Auto-patched remote model code for 100% compatibility.")
                except Exception:
                    pass

def init_model_and_rag():
    global tokenizer, model, vector_store, embeddings_model
    
    print("🚀 Initializing HuggingFace Embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # Load FAISS Vector Store if previously saved to Drive
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            print(f"📚 Loading existing FAISS index from {VECTOR_STORE_PATH}...")
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings_model, 
                allow_dangerous_deserialization=True
            )
            print("✅ Vector Store loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}. Starting fresh.")
            vector_store = None
    else:
        print("ℹ️ No existing vector index found. Ready to accept uploads.")

    # Check if model exists locally in Google Drive with config.json AND weight files
    has_config = os.path.isfile(os.path.join(MODEL_STORAGE_PATH, "config.json"))
    has_weights = any(os.path.exists(os.path.join(MODEL_STORAGE_PATH, f)) for f in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"])
    is_valid_local = os.path.exists(MODEL_STORAGE_PATH) and has_config and has_weights
    target_model_path = MODEL_STORAGE_PATH if is_valid_local else MODEL_NAME
    print(f"🤗 Loading FinanceParam model from: {target_model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    try:
        tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    except Exception as tok_err:
        print(f"⚠️ Could not load tokenizer from {target_model_path}: {tok_err}. Fallback loading tokenizer from HuggingFace Hub: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Auto-patch cached remote code file
    patch_remote_code_if_needed()

    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    model_config.use_cache = False  # Disable KV cache wrapping for legacy model architecture compatibility
    if hasattr(model_config, "rope_scaling") and isinstance(model_config.rope_scaling, dict):
        r_type = model_config.rope_scaling.get("type") or model_config.rope_scaling.get("rope_type")
        if r_type in ["default", None]:
            model_config.rope_scaling = {"type": "linear", "factor": 1.0}
        else:
            model_config.rope_scaling["type"] = r_type if r_type in ["linear", "dynamic"] else "linear"
            if "factor" not in model_config.rope_scaling:
                model_config.rope_scaling["factor"] = 1.0
    else:
        model_config.rope_scaling = {"type": "linear", "factor": 1.0}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as mdl_err:
        print(f"⚠️ Could not load model from {target_model_path}: {mdl_err}. Reloading model from HuggingFace Hub: {MODEL_NAME}")
        target_model_path = MODEL_NAME
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
    
    if device != "cuda":
        model = model.to(device)

    # Force transformers generation engine to use native legacy tuple cache instead of DynamicCache
    model._supports_cache_class = False
    if hasattr(model, "config"):
        model.config._supports_cache_class = False

    # Intercept model.forward to convert any incoming DynamicCache into a legacy tuple or None
    orig_forward = model.forward
    def safe_forward(*args, **kwargs):
        pkv = kwargs.get("past_key_values", None)
        if pkv is None and len(args) > 1:
            pkv = args[1]
        
        if pkv is not None:
            is_empty = False
            if hasattr(pkv, "key_cache"):
                is_empty = (len(pkv.key_cache) == 0)
            elif hasattr(pkv, "__len__"):
                is_empty = (len(pkv) == 0)

            if is_empty:
                if "past_key_values" in kwargs:
                    kwargs["past_key_values"] = None
                elif len(args) > 1:
                    args = (args[0], None) + args[2:]
            elif hasattr(pkv, "to_legacy_cache"):
                leg = pkv.to_legacy_cache()
                if "past_key_values" in kwargs:
                    kwargs["past_key_values"] = leg
            elif hasattr(pkv, "key_cache"):
                leg = tuple(zip(pkv.key_cache, pkv.value_cache))
                if "past_key_values" in kwargs:
                    kwargs["past_key_values"] = leg
            elif type(pkv).__name__ == "DynamicCache":
                if "past_key_values" in kwargs:
                    kwargs["past_key_values"] = None
        return orig_forward(*args, **kwargs)

    model.forward = safe_forward

    # Re-run patch after model download to guarantee downloaded remote files are patched
    patch_remote_code_if_needed()

    # Save to Google Drive if downloaded from HuggingFace Hub
    if target_model_path == MODEL_NAME and os.path.exists(BASE_DIR):
        try:
            print(f"💾 Saving model to Google Drive for fast future restarts: {MODEL_STORAGE_PATH}")
            if hasattr(model, "config"):
                if not hasattr(model.config, "rope_scaling") or not isinstance(model.config.rope_scaling, dict):
                    model.config.rope_scaling = {"type": "linear", "factor": 1.0}
            if hasattr(model, "generation_config"):
                if hasattr(model.generation_config, "rope_scaling") and not isinstance(model.generation_config.rope_scaling, dict):
                    model.generation_config.rope_scaling = None
            model.save_pretrained(MODEL_STORAGE_PATH, safe_serialization=True)
            tokenizer.save_pretrained(MODEL_STORAGE_PATH)
            print("✅ Model saved to Google Drive successfully.")
        except Exception as save_err:
            print(f"⚠️ Note: Could not save model copy to Drive: {save_err}")

    print(f"✅ Model successfully loaded on: {model.device} with dtype: {dtype}")

@app.on_event("startup")
def startup_event():
    global model
    if model is None:
        print("🔄 Running model & RAG initialization on startup...")
        init_model_and_rag()

# Also run init directly on module load if possible
try:
    init_model_and_rag()
except Exception as e:
    import traceback
    print(f"⚠️ Initial module load setup deferred to startup handler: {e}")
    traceback.print_exc()

# Request Models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []
    mode: Optional[str] = "RAG Document Q&A"
    system_prompt: Optional[str] = ""
    temperature: Optional[float] = 0.7
    max_new_tokens: Optional[int] = 512
    top_k: Optional[int] = 4

@app.get("/health")
def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "vram_used_gb": torch.cuda.memory_allocated(0) / (1024**3),
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    else:
        gpu_info = {"available": False, "name": "CPU"}

    indexed_count = 0
    if vector_store is not None and hasattr(vector_store, "index"):
        indexed_count = vector_store.index.ntotal

    return {
        "status": "online",
        "model_loaded": model is not None,
        "vector_store_indexed_chunks": indexed_count,
        "gpu": gpu_info
    }

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    global vector_store, embeddings_model
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    documents = []
    saved_file_paths = []

    for file in files:
        file_path = os.path.join(TEMP_UPLOADS_PATH, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_file_paths.append(file_path)

        # Parse file based on extension
        ext = os.path.splitext(file.filename)[-1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
            elif ext in [".txt", ".md"]:
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error parsing file {file.filename}: {e}")

    if not documents:
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded files.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

    # Update or Create FAISS Vector Store
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings_model)
    else:
        vector_store.add_documents(chunks)

    # Save Vector Store to Google Drive persistent path
    try:
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"💾 Saved FAISS index to {VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save vector store to disk: {e}")

    return {
        "status": "success",
        "documents_processed": len(files),
        "chunks_added": len(chunks),
        "total_chunks": vector_store.index.ntotal if hasattr(vector_store, "index") else len(chunks)
    }

@app.post("/chat")
def chat(request: ChatRequest):
    global model, tokenizer, vector_store
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading or unavailable.")

    try:
        user_message = request.message
        sources = []
        context_str = ""

        # RAG Context Retrieval if vector store exists and mode uses RAG
        if vector_store is not None and request.mode in ["RAG Document Q&A", "Institutional Equity Analysis"]:
            try:
                docs = vector_store.similarity_search(user_message, k=request.top_k)
                retrieved_chunks = []
                for doc in docs:
                    source_name = doc.metadata.get("source", "Document")
                    text = doc.page_content.strip()
                    retrieved_chunks.append(f"[{source_name}]: {text}")
                    sources.append({"source": os.path.basename(source_name), "text": text})

                if retrieved_chunks:
                    context_str = "\n\n--- RELEVANT FINANCIAL CONTEXT ---\n" + "\n\n".join(retrieved_chunks) + "\n-----------------------------------\n"
            except Exception as rag_err:
                print(f"⚠️ RAG search error: {rag_err}")

        # Build Prompt Payload
        prompt_content = user_message
        if context_str:
            prompt_content = f"{context_str}\nUser Question: {user_message}"

        messages = []
        system_prompt = request.system_prompt if request.system_prompt else "You are an expert AI financial analyst specializing in equity research, valuation, and corporate finance."
        messages.append({"role": "system", "content": system_prompt})

        # Append Chat History (Filter out previous error messages)
        for msg in request.history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                content = str(msg["content"])
                if not content.startswith("API Error") and not content.startswith("❌") and not content.startswith("Failed to"):
                    messages.append({"role": msg["role"], "content": content})

        # Append Current Query with RAG context
        messages.append({"role": "user", "content": prompt_content})

        # Safe & Native Prompt Construction (uses model chat_template if available)
        raw_prompt = ""
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            try:
                raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as tmpl_err:
                print(f"⚠️ apply_chat_template fallback: {tmpl_err}")
                raw_prompt = ""

        if not raw_prompt:
            raw_prompt = f"System: {system_prompt}\n\n"
            for m in messages[1:]:
                role_label = "Human" if m["role"] == "user" else "Assistant"
                raw_prompt += f"{role_label}: {m['content']}\n\n"
            raw_prompt += "Assistant:"

        # Tokenization with Attention Mask & Safe Token Clamping
        encoded = tokenizer(raw_prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device) if hasattr(encoded, "attention_mask") and encoded.attention_mask is not None else None

        vocab_size = getattr(model.config, "vocab_size", 32000)
        if not isinstance(vocab_size, int):
            vocab_size = 32000

        # Clamp only if out-of-bounds token IDs are present and input_ids is a Tensor
        if hasattr(input_ids, "dtype") and isinstance(vocab_size, int):
            try:
                if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                    input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
            except Exception:
                pass

        # Pad & EOS Token ID sanitization
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None or not isinstance(pad_id, int) or pad_id >= vocab_size or pad_id < 0:
            pad_id = tokenizer.eos_token_id if (isinstance(tokenizer.eos_token_id, int) and 0 <= tokenizer.eos_token_id < vocab_size) else 0

        # Generation Parameters (with Attention Mask, Repetition Penalty, Top-P to prevent garbled text/loops)
        gen_kwargs = {
            "max_new_tokens": request.max_new_tokens or 512,
            "use_cache": True,
            "pad_token_id": pad_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "top_p": 0.9
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        temp = float(request.temperature) if request.temperature is not None else 0.7
        if temp > 0.01:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
        else:
            gen_kwargs["do_sample"] = False

        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)
            input_length = input_ids.shape[-1]
            response_ids = output_ids[0][input_length:]

        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        if "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()
        inference_time = time.time() - start_time

        return {
            "response": response_text,
            "sources": sources,
            "inference_time_seconds": round(inference_time, 2)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error during chat generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")

@app.get("/documents")
def get_documents():
    if vector_store is None:
        return {"total_chunks": 0, "message": "No vector store index initialized."}
    
    return {
        "total_chunks": vector_store.index.ntotal if hasattr(vector_store, "index") else 0,
        "storage_path": VECTOR_STORE_PATH
    }

@app.post("/clear")
def clear_vector_store():
    global vector_store
    vector_store = None
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            shutil.rmtree(VECTOR_STORE_PATH)
        except Exception as e:
            print(f"Error removing vector store directory: {e}")
    return {"status": "success", "message": "Vector store reset successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
