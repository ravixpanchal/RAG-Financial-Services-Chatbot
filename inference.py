from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_name = os.path.join(os.getcwd(), "FinanceParam")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

chat_history = []

print("FinanceParam Chatbot (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(
        {"role": "user", "content": user_input}
    )

    inputs = tokenizer.apply_chat_template(
        chat_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            repetition_penalty=1.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

    print(f"\nAssistant: {response}\n")

    chat_history.append(
        {"role": "assistant", "content": response}
    )
