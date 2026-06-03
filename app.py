import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Path to the locally cloned model
model_path = os.path.join(os.getcwd(), "FinanceParam")

print(f"Loading model from {model_path}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model
# Use float32 for CPU stability or bfloat16 if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto"
)

print(f"Model loaded on: {model.device}")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    history = data.get("history", [])

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Build prompt with history
    prompt = history + [{"role": "user", "content": user_input}]
    
    # Handle chat template return type (Tensor or BatchEncoding)
    inputs = tokenizer.apply_chat_template(
        prompt, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "use_cache": True
    }

    print(f"Generating response for: {user_input}")
    with torch.no_grad():
        if isinstance(inputs, torch.Tensor):
            output = model.generate(input_ids=inputs, **gen_kwargs)
        else:
            output = model.generate(**inputs, **gen_kwargs)

    # Decode response
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's part (everything after the user's input)
    # Since skip_special_tokens=True removes the tags, we might need a better way to extract.
    # But usually, it returns the whole conversation.
    # A simple way for this model is to split or just return the whole thing and let the front-end handle it,
    # but cleaning it here is better.
    
    # In many models, the assistant response starts after the last user message.
    # Let's find the last occurrence of the user input or just clean up based on known tags if they were preserved.
    # However, since we decoded with skip_special_tokens=True, we have plain text.
    
    response_text = full_text.split("assistant")[-1].strip() if "assistant" in full_text else full_text
    
    # If the above split is too aggressive, we'll just return the full text for now.
    # Actually, let's just return the last part of the generation.
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5566)
