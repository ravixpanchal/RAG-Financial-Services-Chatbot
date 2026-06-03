import os
import torch
from celery import shared_task
from django.conf import settings
from .models import ChatMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load settings
BASE_DIR = settings.BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "FinanceParam")

# Global Initialization of Embeddings
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Global model and tokenizer
_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading FinanceParam model from {MODEL_PATH}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        max_memory = {0: "14GB", "cpu": "40GB"} 
        
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            offload_folder="offload"
        )
    return _model, _tokenizer


def is_response_coherent(text):
    """Check if generated text is coherent enough to return to the user."""
    if not text or len(text.strip()) < 5:
        return False, "empty"
    
    words = text.split()
    
    # Too short
    if len(words) < 3:
        return False, "too_short"
    
    # High repetition = word soup
    if len(words) > 8:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.30:
            return False, f"repetitive (unique={unique_ratio:.0%})"
    
    # Excessive punctuation noise
    if len(text) > 30:
        punct_ratio = sum(1 for c in text if c in '-,.:;') / len(text)
        if punct_ratio > 0.25:
            return False, f"punctuation_noise ({punct_ratio:.0%})"
    
    # Check for hallucinated role-play (model thinks it's someone else)
    bad_patterns = ['chief', 'dear sir', 'dear madam', 'updated version of your',
                    'please advise us', 'we need to create', 'cybercity']
    text_lower = text.lower()
    for pattern in bad_patterns:
        if pattern in text_lower:
            return False, f"hallucination ({pattern})"
    
    return True, "ok"


GREETING_WORDS = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good evening',
                  'good afternoon', 'namaste', 'howdy', 'sup', 'yo', 'hii', 'hiii',
                  'help', 'help me', 'can you help me', 'assistance'}

GREETING_RESPONSE = (
    "Hello! I'm BharatGen FinanceParam Chatbot, your institutional-grade financial analysis assistant. "
    "I can help you with:\n\n"
    "• **Equity Research** — P/E ratios, valuations, peer comparisons\n"
    "• **Financial Calculations** — ROE, margins, CAGR, DCF\n"
    "• **Indian Markets** — SEBI regulations, GST, RBI policies\n"
    "• **Risk Analysis** — Portfolio risks, sector analysis\n"
    "• **Document Analysis** — Upload PDFs/Excel files for AI-powered insights\n\n"
    "How can I assist you today?"
)


@shared_task
def process_finance_query(user_input, chat_history_str, system_prompt=''):
    try:
        user_stripped = user_input.lower().strip()
        
        # ── 0. Greetings: instant response, no model call ──
        if user_stripped in GREETING_WORDS:
            chat_msg = ChatMessage.objects.create(
                user_message=user_input, bot_response=GREETING_RESPONSE
            )
            return {'response': GREETING_RESPONSE, 'message_id': chat_msg.id}
        
        # ── 0.5. Short vague query guard ──
        # If the query is very short and not a greeting, the 2B model will hallucinate.
        if len(user_stripped) < 15 and "?" not in user_stripped:
            fallback = "Could you please provide more details or ask a specific financial question? For example: 'What is P/E ratio?' or 'Explain GST'."
            chat_msg = ChatMessage.objects.create(
                user_message=user_input, bot_response=fallback
            )
            return {'response': fallback, 'message_id': chat_msg.id}
        
        # ── 1. RAG retrieval (only for substantive queries) ──
        rag_context = ""
        if len(user_stripped) > 10:  # Skip RAG for very short inputs
            try:
                persist_dir = os.path.join(BASE_DIR, 'chroma_db')
                vectorstore = Chroma(persist_directory=persist_dir, embedding_function=_embeddings)
                docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(user_input)
                if docs:
                    rag_context = "\n".join([doc.page_content[:500] for doc in docs[:2]])
            except Exception as e:
                print(f"[RAG] Retrieval failed (non-fatal): {e}")
        
        model, tokenizer = get_model()
        
        # ── 2. Build prompt — KEEP IT MINIMAL for 2B model ──
        # The key insight: this model works best with SHORT, direct prompts.
        # Do NOT inject the full system prompt — it destroys coherence.
        
        if rag_context.strip():
            # Document Q&A mode
            user_content = (
                f"Based on this document:\n{rag_context[:800]}\n\n"
                f"Answer: {user_input}"
            )
        else:
            # Direct Q&A mode — just the question, nothing else
            user_content = user_input
        
        # ── 3. Apply chat template ──
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # ── 4. Tokenize with strict limit ──
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[1]
        print(f"[FinanceParam] Input: {input_len} tokens | Query: '{user_input[:50]}'")
        
        # ── 5. Generate ──
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        # ── 6. Decode ──
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if "<|assistant|>" in full_output:
            bot_resp = full_output.split("<|assistant|>")[-1]
            for tag in ["<|/assistant|>", "</s>", "<|user|>", "<|/user|>", "<pad>", "<s>"]:
                bot_resp = bot_resp.replace(tag, "")
            bot_resp = bot_resp.strip()
        else:
            bot_resp = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # ── 7. Quality check ──
        is_ok, reason = is_response_coherent(bot_resp)
        
        if not is_ok:
            print(f"[FinanceParam] ⚠️ Response failed quality check: {reason}")
            print(f"[FinanceParam] Raw output: {bot_resp[:200]}")
            
            # Provide a helpful fallback instead of garbage
            bot_resp = (
                f"I understand you're asking about: *\"{user_input}\"*\n\n"
                "I wasn't able to generate a clear analysis for this query. "
                "For best results with this model, try:\n\n"
                "• **Specific questions**: \"What is the P/E ratio?\"\n"
                "• **Definitions**: \"Explain EBITDA\"\n"
                "• **Calculations**: \"Calculate ROE given net income of 10 crore and equity of 50 crore\"\n"
                "• **Document queries**: Upload a financial document first, then ask about it\n\n"
                "Please try rephrasing your question."
            )
        
        # ── 8. Save & return ──
        chat_msg = ChatMessage.objects.create(
            user_message=user_input,
            bot_response=bot_resp
        )
        
        return {'response': bot_resp, 'message_id': chat_msg.id}
        
    except Exception as e:
        print(f"Error in process_finance_query: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


@shared_task
def index_documents(file_data_list):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import pypdf
        import docx
        import pandas as pd
        
        persist_dir = os.path.join(BASE_DIR, 'chroma_db')
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=_embeddings)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = []

        for file_info in file_data_list:
            path = file_info['path']
            name = file_info['name']
            content = ""
            ext = os.path.splitext(name)[1].lower()
            
            try:
                if ext == '.pdf':
                    with open(path, 'rb') as f:
                        reader = pypdf.PdfReader(f)
                        for page in reader.pages:
                            content += page.extract_text() + "\n"
                elif ext == '.docx':
                    doc = docx.Document(path)
                    for para in doc.paragraphs:
                        content += para.text + "\n"
                elif ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(path)
                    content = df.to_string()
                elif ext == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                if content.strip():
                    chunks = text_splitter.create_documents([content], metadatas=[{"source": name}])
                    all_chunks.extend(chunks)
                
                if os.path.exists(path):
                    os.remove(path)
            except Exception as fe:
                print(f"Error processing file {name}: {fe}")

        if all_chunks:
            vectorstore.add_documents(all_chunks)
            return {'status': 'success', 'count': len(all_chunks)}
        else:
            return {'status': 'error', 'message': 'No text extracted'}

    except Exception as e:
        print(f"Error in index_documents: {e}")
        return {'status': 'error', 'message': str(e)}
