import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Path to the locally cloned model
model_path = os.path.join(os.getcwd(), "FinanceParam")

print(f"Loading model from {model_path}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32, # CPU friendly
    device_map="auto"
)

print(f"Model loaded on: {model.device}")

# Example Finance query
user_input = "How to file income tax return in India? Tell me in detail."

# Prepare prompt
prompt = [{"role": "user", "content": user_input}]
inputs = tokenizer.apply_chat_template(
    prompt, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to(model.device)

print("Generating response...")
with torch.no_grad():
    # Handle different return types of apply_chat_template
    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.7,
        "use_cache": True
    }
    if isinstance(inputs, torch.Tensor):
        output = model.generate(input_ids=inputs, **gen_kwargs)
    else:
        output = model.generate(**inputs, **gen_kwargs)


print("\n--- Model Output (Decoded) ---\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
