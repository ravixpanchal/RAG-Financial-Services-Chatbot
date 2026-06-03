#!/usr/bin/env python3
"""
BharatGen FinanceParam — Prompt Optimization Tester
Tests different prompt strategies to find the optimal approach for the 2B model.

Strategies tested:
  1. No system prompt (raw question only)
  2. Minimal system prompt (1 line)
  3. Concise system prompt (~5 lines)
  4. Full prompt.txt (the current default)
  5. Instruction-in-question format (no system role)

Usage:
    cd /home/bisagn/Desktop/BharatGenFinance
    ./venv/bin/python benchmarks/prompt_optimization.py
"""

import os
import sys
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FinanceParam")

# ──────────────────────────────────────────────
# Prompt Strategies
# ──────────────────────────────────────────────
STRATEGIES = {
    "no_prompt": {
        "name": "No System Prompt",
        "description": "Raw question only, no system context",
        "build": lambda q: q,
    },
    "minimal": {
        "name": "Minimal Prompt",
        "description": "One-line role assignment",
        "build": lambda q: f"You are a financial analyst. {q}",
    },
    "concise": {
        "name": "Concise Prompt",
        "description": "Short, focused instructions",
        "build": lambda q: (
            "You are an expert financial analyst. Answer the following question clearly and accurately. "
            "Be specific, use data when available, and keep your response professional.\n\n"
            f"Question: {q}"
        ),
    },
    "structured": {
        "name": "Structured Prompt",
        "description": "Question with explicit output format",
        "build": lambda q: (
            f"Question: {q}\n\n"
            "Please provide a clear, well-structured answer. "
            "Use bullet points if listing multiple items. "
            "Include relevant numbers and facts where applicable."
        ),
    },
    "full_prompt": {
        "name": "Full prompt.txt",
        "description": "The complete institutional analysis prompt",
        "build": None,  # Will be set at runtime
    },
}

# Test questions used across all strategies
TEST_QUESTIONS = [
    {
        "id": "PO-1",
        "question": "What is the P/E ratio and why is it important?",
        "keywords": ["price", "earnings", "valuation", "stock", "ratio"],
    },
    {
        "id": "PO-2",
        "question": "What are the key financial risks for Indian banking sector?",
        "keywords": ["NPA", "risk", "credit", "interest", "regulation", "bank"],
    },
    {
        "id": "PO-3",
        "question": "A company has revenue of 1000 crore and net profit of 100 crore. Calculate the net profit margin.",
        "keywords": ["10", "percent", "margin", "profit"],
    },
]


def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "14GB", "cpu": "40GB"},
        offload_folder="offload",
    )
    return model, tokenizer


def generate(model, tokenizer, user_content, max_new_tokens=256):
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>")[-1]
        for tag in ["<|/assistant|>", "</s>", "<|user|>", "<|/user|>", "<pad>", "<s>"]:
            response = response.replace(tag, "")
        response = response.strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response, elapsed, input_len


def evaluate_quality(response, keywords):
    """Evaluate response quality on multiple dimensions."""
    response_lower = response.lower()
    words = response.split()
    
    # 1. Keyword score
    hits = [kw for kw in keywords if kw.lower() in response_lower]
    keyword_score = len(hits) / len(keywords) if keywords else 0
    
    # 2. Coherence score (penalize repetition, word soup)
    coherence = 1.0
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            coherence = 0.1
        elif unique_ratio < 0.5:
            coherence = 0.5
    
    if len(words) < 5:
        coherence *= 0.3
    
    # 3. Punctuation noise
    if len(response) > 30:
        punct_ratio = sum(1 for c in response if c in "-,.:;") / len(response)
        if punct_ratio > 0.2:
            coherence *= 0.3
    
    # 4. Response length appropriateness
    length_score = 1.0
    if len(words) < 10:
        length_score = 0.3
    elif len(words) > 500:
        length_score = 0.7  # Overly verbose
    
    overall = (keyword_score * 0.4 + coherence * 0.4 + length_score * 0.2)
    
    return {
        "keyword_score": round(keyword_score, 2),
        "coherence": round(coherence, 2),
        "length_score": round(length_score, 2),
        "overall": round(overall, 2),
        "word_count": len(words),
        "unique_ratio": round(len(set(words)) / len(words), 2) if words else 0,
        "keyword_hits": hits,
    }


def run_optimization():
    model, tokenizer = load_model()
    
    # Load full prompt.txt
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompt.txt")
    with open(prompt_path, "r") as f:
        full_prompt = f.read()
    
    # Set the full_prompt strategy builder
    STRATEGIES["full_prompt"]["build"] = lambda q: f"{full_prompt}\n\nQuestion: {q}"
    
    # Count tokens for each strategy
    print("\n" + "═" * 80)
    print("  PROMPT OPTIMIZATION TEST")
    print("═" * 80)
    
    results = {}
    
    for strategy_key, strategy in STRATEGIES.items():
        print(f"\n{'━' * 60}")
        print(f"  Strategy: {strategy['name']}")
        print(f"  {strategy['description']}")
        print(f"{'━' * 60}")
        
        strategy_results = []
        total_score = 0.0
        
        for tq in TEST_QUESTIONS:
            user_content = strategy["build"](tq["question"])
            
            print(f"\n  [{tq['id']}] {tq['question']}")
            
            try:
                response, elapsed, input_tokens = generate(model, tokenizer, user_content)
                quality = evaluate_quality(response, tq["keywords"])
                
                display = response[:200] + "..." if len(response) > 200 else response
                icon = "✅" if quality["overall"] >= 0.5 else "⚠️" if quality["overall"] > 0.2 else "❌"
                
                print(f"  {icon} Quality: {quality['overall']:.0%} (kw:{quality['keyword_score']:.0%} coh:{quality['coherence']:.0%})")
                print(f"     Input tokens: {input_tokens} | Words: {quality['word_count']} | Time: {elapsed:.1f}s")
                print(f"     Response: {display}")
                
                strategy_results.append({
                    "question_id": tq["id"],
                    "input_tokens": input_tokens,
                    "quality": quality,
                    "elapsed_sec": round(elapsed, 2),
                    "response_preview": response[:300],
                })
                total_score += quality["overall"]
                
            except Exception as e:
                print(f"  💥 ERROR: {e}")
                strategy_results.append({"question_id": tq["id"], "error": str(e), "quality": {"overall": 0}})
        
        avg_score = total_score / len(TEST_QUESTIONS)
        avg_tokens = sum(r.get("input_tokens", 0) for r in strategy_results) / len(strategy_results)
        
        results[strategy_key] = {
            "name": strategy["name"],
            "avg_score": round(avg_score, 3),
            "avg_input_tokens": round(avg_tokens),
            "details": strategy_results,
        }
        
        print(f"\n  ► Strategy Average Score: {avg_score:.0%} (avg input: {avg_tokens:.0f} tokens)")
    
    # ──────────────────────────────────────
    # Comparison Table
    # ──────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STRATEGY COMPARISON")
    print("═" * 80)
    print(f"\n  {'Strategy':<25} {'Avg Score':>10} {'Avg Tokens':>12} {'Recommendation':>20}")
    print(f"  {'─' * 67}")
    
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)
    
    for i, (key, data) in enumerate(sorted_strategies):
        rec = "⭐ BEST" if i == 0 else "GOOD" if data["avg_score"] >= 0.5 else "AVOID"
        print(f"  {data['name']:<25} {data['avg_score']:>9.0%} {data['avg_input_tokens']:>10.0f} {rec:>20}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"prompt_optimization_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved to: {output_path}")
    
    # Recommendation
    best_key = sorted_strategies[0][0]
    best = sorted_strategies[0][1]
    print(f"\n  ✅ RECOMMENDED STRATEGY: '{best['name']}' (Score: {best['avg_score']:.0%}, Tokens: {best['avg_input_tokens']:.0f})")
    
    return results


if __name__ == "__main__":
    run_optimization()
