#!/usr/bin/env python3
"""
BharatGen FinanceParam — Financial Accuracy Benchmark Suite
Evaluates the model's financial knowledge with structured, auto-scored tests.

Categories:
  1. Financial Definitions (multiple choice)
  2. Numerical Calculations (exact answer)
  3. Indian Finance Knowledge (multiple choice)
  4. Risk & Valuation Reasoning (keyword scoring)
  5. Regulatory Knowledge (multiple choice)

Usage:
    cd /home/bisagn/Desktop/BharatGenFinance
    ./venv/bin/python benchmarks/financial_accuracy_benchmark.py
"""

import os
import sys
import re
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FinanceParam")

# ──────────────────────────────────────────────
# BENCHMARK DATASET
# ──────────────────────────────────────────────

BENCHMARKS = {
    "financial_definitions": {
        "description": "Basic financial term definitions",
        "questions": [
            {
                "id": "FD-01",
                "question": "What does ROE stand for in finance? Answer with the full form only.",
                "correct_answer": "Return on Equity",
                "scoring": "exact_match",
                "keywords": ["return", "equity"],
            },
            {
                "id": "FD-02",
                "question": "What is EBITDA? Give the full form.",
                "correct_answer": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
                "scoring": "keyword",
                "keywords": ["earnings", "interest", "tax", "depreciation", "amortization"],
            },
            {
                "id": "FD-03",
                "question": "What is the difference between a stock and a bond? Answer in 2 sentences.",
                "correct_answer": "A stock represents ownership in a company while a bond represents a loan to a company or government.",
                "scoring": "keyword",
                "keywords": ["ownership", "loan", "debt", "equity", "company"],
            },
            {
                "id": "FD-04",
                "question": "Define 'market capitalization' in one sentence.",
                "correct_answer": "Market capitalization is the total market value of a company's outstanding shares.",
                "scoring": "keyword",
                "keywords": ["market", "value", "shares", "outstanding", "price"],
            },
            {
                "id": "FD-05",
                "question": "What is a mutual fund? Answer briefly.",
                "correct_answer": "A mutual fund is a pooled investment vehicle managed by professionals.",
                "scoring": "keyword",
                "keywords": ["pool", "invest", "fund", "manager", "diversif"],
            },
        ],
    },
    "numerical_calculations": {
        "description": "Financial calculations requiring numerical answers",
        "questions": [
            {
                "id": "NC-01",
                "question": "A company has net income of 10 crore and total equity of 50 crore. What is the ROE? Give only the percentage.",
                "correct_answer": "20%",
                "scoring": "numeric",
                "expected_value": 20.0,
                "tolerance": 1.0,
            },
            {
                "id": "NC-02",
                "question": "A stock is trading at Rs 500 per share and its EPS is Rs 25. What is the P/E ratio? Give only the number.",
                "correct_answer": "20",
                "scoring": "numeric",
                "expected_value": 20.0,
                "tolerance": 0.5,
            },
            {
                "id": "NC-03",
                "question": "A company has revenue of Rs 1000 crore and net profit of Rs 150 crore. What is the net profit margin in percentage?",
                "correct_answer": "15%",
                "scoring": "numeric",
                "expected_value": 15.0,
                "tolerance": 1.0,
            },
            {
                "id": "NC-04",
                "question": "If a company has current assets of Rs 200 crore and current liabilities of Rs 100 crore, what is the current ratio?",
                "correct_answer": "2",
                "scoring": "numeric",
                "expected_value": 2.0,
                "tolerance": 0.1,
            },
            {
                "id": "NC-05",
                "question": "A bond has a face value of Rs 1000, pays annual coupon of Rs 80. What is the coupon rate in percentage?",
                "correct_answer": "8%",
                "scoring": "numeric",
                "expected_value": 8.0,
                "tolerance": 0.5,
            },
        ],
    },
    "indian_finance": {
        "description": "India-specific financial knowledge",
        "questions": [
            {
                "id": "IF-01",
                "question": "What is the full form of SEBI?",
                "correct_answer": "Securities and Exchange Board of India",
                "scoring": "keyword",
                "keywords": ["securities", "exchange", "board", "india"],
            },
            {
                "id": "IF-02",
                "question": "What is the current GST structure in India? List the main slabs.",
                "correct_answer": "0%, 5%, 12%, 18%, 28%",
                "scoring": "keyword",
                "keywords": ["5", "12", "18", "28"],
            },
            {
                "id": "IF-03",
                "question": "What is the Nifty 50 index?",
                "correct_answer": "The Nifty 50 is a stock market index of 50 of the largest Indian companies listed on the National Stock Exchange.",
                "scoring": "keyword",
                "keywords": ["50", "index", "stock", "national", "exchange", "india"],
            },
            {
                "id": "IF-04",
                "question": "What is a Demat account in India?",
                "correct_answer": "A Demat account holds securities in electronic form in India.",
                "scoring": "keyword",
                "keywords": ["electronic", "securities", "shares", "dematerial"],
            },
            {
                "id": "IF-05",
                "question": "What is Section 80C of the Indian Income Tax Act?",
                "correct_answer": "Section 80C allows deductions up to Rs 1.5 lakh for specified investments and expenses.",
                "scoring": "keyword",
                "keywords": ["deduction", "1.5", "lakh", "tax", "invest"],
            },
        ],
    },
    "risk_valuation": {
        "description": "Risk assessment and valuation reasoning",
        "questions": [
            {
                "id": "RV-01",
                "question": "What are the key risks of investing in small-cap stocks? List at least 3 risks.",
                "correct_answer": "Liquidity risk, higher volatility, limited information, business risk, regulatory risk",
                "scoring": "keyword",
                "keywords": ["liquidity", "volatil", "risk", "small", "information"],
            },
            {
                "id": "RV-02",
                "question": "Explain the concept of beta in stock market investing.",
                "correct_answer": "Beta measures a stock's volatility relative to the overall market.",
                "scoring": "keyword",
                "keywords": ["volatil", "market", "risk", "systematic", "1"],
            },
            {
                "id": "RV-03",
                "question": "What is the difference between fundamental and technical analysis?",
                "correct_answer": "Fundamental analysis examines financial statements and business value; technical analysis studies price patterns and charts.",
                "scoring": "keyword",
                "keywords": ["fundamental", "technical", "price", "financial", "chart", "value"],
            },
            {
                "id": "RV-04",
                "question": "Why is diversification important in portfolio management?",
                "correct_answer": "Diversification reduces unsystematic risk by spreading investments across different assets.",
                "scoring": "keyword",
                "keywords": ["risk", "spread", "different", "asset", "reduce"],
            },
            {
                "id": "RV-05",
                "question": "What is DCF valuation?",
                "correct_answer": "Discounted Cash Flow valuation estimates the present value of expected future cash flows.",
                "scoring": "keyword",
                "keywords": ["discount", "cash", "flow", "future", "present", "value"],
            },
        ],
    },
    "coherence_stress": {
        "description": "Tests model coherence under longer prompts",
        "questions": [
            {
                "id": "CS-01",
                "question": "Write a brief investment thesis for Reliance Industries in 3 bullet points.",
                "correct_answer": "N/A",
                "scoring": "coherence",
                "keywords": ["reliance", "oil", "jio", "retail", "digital", "energy"],
            },
            {
                "id": "CS-02",
                "question": "Compare SBI and HDFC Bank as investment options. Give 2 pros and 2 cons for each.",
                "correct_answer": "N/A",
                "scoring": "coherence",
                "keywords": ["SBI", "HDFC", "bank", "loan", "NPA", "growth", "private", "public"],
            },
        ],
    },
}


def load_model():
    """Load model and tokenizer."""
    print(f"Loading model from: {MODEL_PATH}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "14GB", "cpu": "40GB"},
        offload_folder="offload",
    )
    print(f"Model loaded in {time.time() - start:.1f}s")
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=256):
    """Generate a response."""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    
    return response, elapsed


def score_keyword(response, keywords):
    """Score based on keyword presence."""
    response_lower = response.lower()
    hits = [kw for kw in keywords if kw.lower() in response_lower]
    score = len(hits) / len(keywords) if keywords else 0
    return score, hits


def score_numeric(response, expected, tolerance):
    """Extract a number from the response and compare."""
    numbers = re.findall(r'[\d]+\.?[\d]*', response)
    if not numbers:
        return 0.0, None
    
    # Try to find the closest match
    for num_str in numbers:
        try:
            val = float(num_str)
            if abs(val - expected) <= tolerance:
                return 1.0, val
        except ValueError:
            continue
    
    # Partial credit if any number is close
    closest = min(numbers, key=lambda x: abs(float(x) - expected))
    closest_val = float(closest)
    distance = abs(closest_val - expected)
    if distance <= tolerance * 5:
        return 0.5, closest_val
    return 0.0, closest_val


def score_coherence(response, keywords):
    """Score coherence: keyword presence + readability heuristics."""
    kw_score, hits = score_keyword(response, keywords)
    
    words = response.split()
    penalties = 0
    
    # Penalize very short responses
    if len(words) < 10:
        penalties += 0.3
    
    # Penalize excessive repetition
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            penalties += 0.4
    
    # Penalize excessive punctuation
    if len(response) > 30:
        punct_ratio = sum(1 for c in response if c in "-,.:;") / len(response)
        if punct_ratio > 0.2:
            penalties += 0.3
    
    final_score = max(0, kw_score - penalties)
    return final_score, hits


def run_benchmark():
    """Run the full benchmark suite."""
    model, tokenizer = load_model()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    overall_scores = {}
    
    print("\n" + "═" * 80)
    print("  BHARATGEN FINANCEPARAM — FINANCIAL ACCURACY BENCHMARK")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Timestamp: {timestamp}")
    print("═" * 80)
    
    total_questions = 0
    total_score = 0.0
    
    for category_key, category_data in BENCHMARKS.items():
        print(f"\n{'─' * 60}")
        print(f"  Category: {category_data['description']}")
        print(f"{'─' * 60}")
        
        cat_results = []
        cat_score = 0.0
        
        for q in category_data["questions"]:
            print(f"\n  [{q['id']}] {q['question']}")
            
            try:
                response, elapsed = generate(model, tokenizer, q["question"])
                
                # Score based on type
                if q["scoring"] == "keyword" or q["scoring"] == "exact_match":
                    score, hits = score_keyword(response, q["keywords"])
                elif q["scoring"] == "numeric":
                    score, extracted_val = score_numeric(response, q["expected_value"], q["tolerance"])
                    hits = [f"extracted={extracted_val}"]
                elif q["scoring"] == "coherence":
                    score, hits = score_coherence(response, q["keywords"])
                else:
                    score, hits = 0.0, []
                
                # Detect degradation
                words = response.split()
                degraded = False
                if len(words) > 10:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.3:
                        degraded = True
                        score = min(score, 0.1)
                
                display_response = response[:200] + "..." if len(response) > 200 else response
                status_icon = "✅" if score >= 0.5 else "⚠️" if score > 0 else "❌"
                if degraded:
                    status_icon = "💀"
                
                print(f"  {status_icon} Score: {score:.0%} | Time: {elapsed:.1f}s")
                print(f"     Response: {display_response}")
                if q["scoring"] == "numeric":
                    print(f"     Expected: {q['expected_value']} | Got: {hits}")
                else:
                    print(f"     Keywords matched: {hits}")
                
                cat_results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "expected": q.get("correct_answer", ""),
                    "response": response[:500],
                    "score": score,
                    "elapsed_sec": round(elapsed, 2),
                    "degraded": degraded,
                })
                cat_score += score
                
            except Exception as e:
                print(f"  💥 ERROR: {e}")
                cat_results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "score": 0.0,
                    "error": str(e),
                })
            
            total_questions += 1
        
        avg_cat_score = cat_score / len(category_data["questions"]) if category_data["questions"] else 0
        total_score += cat_score
        overall_scores[category_key] = {
            "avg_score": round(avg_cat_score, 3),
            "total_questions": len(category_data["questions"]),
        }
        all_results[category_key] = cat_results
        
        print(f"\n  Category Score: {avg_cat_score:.0%}")
    
    # ──────────────────────────────────────
    # Final Report
    # ──────────────────────────────────────
    overall_avg = total_score / total_questions if total_questions > 0 else 0
    
    print("\n" + "═" * 80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("═" * 80)
    print(f"\n  {'Category':<30} {'Score':>10} {'Questions':>12}")
    print(f"  {'─' * 52}")
    
    for cat_key, cat_info in overall_scores.items():
        desc = BENCHMARKS[cat_key]["description"][:28]
        print(f"  {desc:<30} {cat_info['avg_score']:>9.0%} {cat_info['total_questions']:>10}")
    
    print(f"  {'─' * 52}")
    print(f"  {'OVERALL':.<30} {overall_avg:>9.0%} {total_questions:>10}")
    
    grade = "A+" if overall_avg >= 0.9 else "A" if overall_avg >= 0.8 else "B" if overall_avg >= 0.7 else "C" if overall_avg >= 0.5 else "D" if overall_avg >= 0.3 else "F"
    print(f"\n  Final Grade: {grade}")
    
    # Save
    report = {
        "model": MODEL_PATH,
        "timestamp": timestamp,
        "overall_score": round(overall_avg, 3),
        "grade": grade,
        "category_scores": overall_scores,
        "detailed_results": all_results,
    }
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"benchmark_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    run_benchmark()
