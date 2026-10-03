---
language:
- en
- hi
tags:
- Multiturn
- QnA
base_model:
- bharatgenai/Param-1-2.9B-Instruct
pipeline_tag: text-generation
library_name: transformers
license: apache-2.0
---

<div align="center">
  <img src="https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct/resolve/main/BharatGen%20Logo%20(1).png" width="60%" alt="BharatGen" />
</div>
<hr>
<div align="center">
  <a href="https://huggingface.co/bharatgenai/FinanceParam" target="_blank" style="margin: 4px;">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-ffda4b?style=flat&logo=huggingface&logoColor=black" />
  </a>
  <a href="https://huggingface.co/bharatgenai/FinanceParam/blob/main/LICENSE" target="_blank" style="margin: 4px;">
    <img alt="License" src="https://img.shields.io/badge/License-yellow.svg" />
  </a>
</div>

# FinanceParam

**BharatGen** introduces **FinanceParam**, a domain-specialized large language model fine-tuned from **Param-1-2.9B-Instruct** on a high-quality finance dataset.
FinanceParam is designed to deliver accurate, bilingual (English-Hindi) Indian financial knowledge for personal finance, taxation, banking, investments, and policy guidance.

---

## 💰 Motivation

Finance touches every aspect of daily life, from household budgeting to national economic policy. Yet, existing language models lack deep domain expertise in Indian finance, regulatory frameworks, and cultural nuances.
**FinanceParam** bridges this gap by combining Param-1’s bilingual capabilities with a meticulously curated financial knowledge base tailored for India.

---

## 🏗 Model Architecture

FinanceParam inherits the architecture of Param-1-2.9B-Instruct:

* **Hidden size**: 2048  
* **Intermediate size**: 7168  
* **Attention heads**: 16  
* **Hidden layers**: 32  
* **Key-value heads**: 8  
* **Max position embeddings**: 2048  
* **Activation**: SiLU  
* **Positional Embeddings**: Rotary (RoPE, theta=10000)  
* **Attention Mechanism**: Grouped-query attention  
* **Precision**: bf16-mixed  
* **Base model**: [Param-1-2.9B-Instruct](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct)

---

## 📚 Data Preparation

FinanceParam’s training corpus was carefully crafted to ensure deep  Indian Finance knowledge, cultural relevance, and bilingual (English-Hindi) accessibility.

**Steps involved:**

1. **Source Gathering**  
   * 10K+ open-source, India-focused finance news & information passages.
   
2. **Question Generation**  
   * Generated 5 curated Q&A pairs per passage using an open-source LLM.

3. **Domain Taxonomy & Personas**  
   * Built an exhaustive, India-specific financial taxonomy.  
   * Defined CA, policy-maker, business and multiple such personas.

4. **Dataset Construction**  
   * 2M Q&A pairs grounded in taxonomy and personas.  
   * Complete dataset translated into Hindi.  
   * 6M multi-turn conversation samples created.


1. **Source Gathering**
    * Collected 25,000+ finance-focused passages from trusted Indian sources: government portals (Income Tax Dept., RBI, SEBI, IRDAI), banking reports, investment advisories, policy documents, and financial news.

2. **Knowledge-Enriched Question Generation**
    * For each passage, an open-source LLM generated 5 high-quality Q&A pairs, refined to cover personal finance, taxation, banking, insurance, and investment topics.

3. **Domain Taxonomy & Personas**
    * Built a comprehensive Indian finance taxonomy spanning income, budgeting, taxation, insurance, banking, and investments.
    * Defined diverse user personas: salaried professionals, students, investors, small business owners, retirees, and policy-makers.

4. **Dataset Construction**
    * Compiled 9M Q&A pairs grounded in taxonomy and personas.
    * Translated the entire dataset into Hindi to ensure accessibility across India’s multilingual audience.
    * Expanded into 8M multi-turn dialogues


---

## 🏋️ Training Setup

* **Base model**: Param-1-2.9B-Instruct  
* **Training framework**: Transformer Framework + `pytorch` multi-node setup  
* **Prompt template**: Custom-designed for financial system inference  
* **Scheduler**: Linear
* **Epochs**: 1
* **Total training samples**: 24M
* **Learning rate**: 2e-4
* **Vocab size**: 256K  
* **Batch size**: 512  

---

## 🚀 Inference Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "bharatgenai/FinanceParam"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.bfloat32,
    device_map="auto"
)

# Example Finance query
user_input = "How to file income tax return. Tell me in detail"

# Based on your requirements use the type of prompt (refere the above examples)
# Append assistant and user for chat model.
prompt = [{"role": "user", "content": user_input}]

inputs = tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=300,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📊 Benchmarks

### Overall BBF Performance
This table shows the **average BBF (Benchmark for Finance)** performance across all tasks, split by **English** and **Hindi** subsets.

| Model                  | BBF   | BBF (English) | BBF (Hindi) |
|-------------------------|-------|---------------|-------------|
| gemma-2-2b-it           | 30.24 | 31.26         | 27.93       |
| Llama-3.2-1B-Instruct   | 26.21 | 26.28         | 26.04       |
| Llama-3.2-3B-Instruct   | 31.76 | 32.94         | 29.09       |
| Qwen2.5-3B-Instruct     | 33.09 | 34.84         | 29.17       |
| granite-3.1-2b-instruct | 31.07 | 32.82         | 27.11       |
| FinanceParam            | 31.42 | 32.24         | 29.56       |

---

### Domain-Wise Performance
This table highlights how models perform across **specific finance-related domains** such as banking, taxation, insurance, economics, etc.

| Domain                                           | gemma-2-2b-it | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Qwen2.5-3B-Instruct | granite-3.1-2b-instruct | FinanceParam |
|--------------------------------------------------|---------------|------------------------|------------------------|---------------------|--------------------------|--------------|
| Accounting                                       | 30.53         | 26.13                 | 27.68                 | 31.82              | 30.92                   | 31.05        |
| Banking Services                                 | 34.67         | 28.18                 | 38.68                 | 36.89              | 34.33                   | 35.78        |
| Behavioral Finance                               | 46.27         | 28.36                 | 37.31                 | 44.78              | 44.78                   | 47.76        |
| Business Management                              | 45.78         | 26.51                 | 53.01                 | 40.96              | 40.96                   | 44.58        |
| Commerce                                         | 31.05         | 27.46                 | 31.52                 | 33.72              | 32.21                   | 28.51        |
| Corporate Finance & Investment                   | 31.98         | 26.37                 | 35.05                 | 37.58              | 31.87                   | 35.05        |
| Data & Analytics in Finance                      | 27.56         | 18.11                 | 20.47                 | 28.35              | 38.58                   | 35.43        |
| Economics & Development Studies                  | 41.24         | 32.85                 | 40.51                 | 44.16              | 37.59                   | 40.88        |
| Energy, Infrastructure & Finance                 | 28.05         | 28.05                 | 39.02                 | 30.49              | 39.02                   | 34.15        |
| Environmental Finance                            | 34.52         | 29.76                 | 38.69                 | 44.05              | 41.67                   | 45.83        |
| Finance Education                                | 39.83         | 25.42                 | 34.75                 | 43.22              | 41.53                   | 31.36        |
| Financial Markets                                | 36.17         | 29.79                 | 48.94                 | 42.55              | 34.04                   | 40.43        |
| Financial Technology                             | 47.83         | 13.04                 | 34.78                 | 39.13              | 34.78                   | 43.48        |
| General Knowledge                                | 38.40         | 28.94                 | 43.04                 | 38.22              | 39.15                   | 40.07        |
| Governance & Policy                              | 34.21         | 27.63                 | 39.29                 | 38.16              | 35.15                   | 38.16        |
| Healthcare Economics                             | 39.47         | 31.58                 | 41.23                 | 45.61              | 34.21                   | 36.84        |
| History, Sociology & Cultural Studies of Finance | 41.73         | 30.71                 | 44.88                 | 38.58              | 37.01                   | 45.67        |
| Information Technology Finance                   | 44.49         | 35.51                 | 53.06                 | 58.16              | 48.16                   | 58.16        |
| Insurance & Risk Management                      | 30.95         | 26.19                 | 38.10                 | 38.10              | 33.33                   | 35.71        |
| Interdisciplinary Finance                        | 36.60         | 30.72                 | 33.33                 | 36.60              | 37.25                   | 37.25        |
| International Finance & Trade                    | 42.17         | 34.94                 | 39.76                 | 42.17              | 36.14                   | 45.78        |
| Language & Communication                         | 40.06         | 29.18                 | 40.59                 | 42.71              | 35.94                   | 41.65        |
| Legal Finance                                    | 41.18         | 20.59                 | 20.59                 | 23.53              | 50.00                   | 20.59        |
| Marketing Finance                                | 35.71         | 38.10                 | 38.10                 | 50.00              | 54.76                   | 61.90        |
| Mathematics for Finance                          | 25.96         | 24.91                 | 27.57                 | 29.85              | 27.66                   | 25.59        |
| Problem Solving                                  | 24.76         | 23.65                 | 25.15                 | 26.20              | 26.56                   | 25.71        |
| Rural Economics                                  | 40.61         | 30.65                 | 44.83                 | 45.21              | 41.76                   | 47.13        |
| Science and Technology in Finance                | 37.62         | 30.69                 | 41.58                 | 43.56              | 27.72                   | 40.59        |
| Sports, Media & Finance Linkages                 | 48.89         | 28.89                 | 42.22                 | 53.33              | 28.89                   | 35.56        |
| Taxation & Regulatory Compliance                 | 45.81         | 31.61                 | 47.10                 | 38.71              | 31.61                   | 37.42        |

---

### Difficulty-Level Performance
This table breaks down performance across **Easy, Medium, and Hard difficulty levels**.

| Difficulty | gemma-2-2b-it | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Qwen2.5-3B-Instruct | granite-3.1-2b-instruct | FinanceParam |
|------------|---------------|------------------------|------------------------|---------------------|--------------------------|--------------|
| Easy       | 36.55         | 28.72                 | 39.73                 | 39.91              | 36.68                   | 38.31        |
| Hard       | 23.20         | 22.43                 | 23.87                 | 25.02              | 25.32                   | 26.60        |
| Medium     | 27.67         | 25.50                 | 28.20                 | 30.48              | 28.63                   | 27.71        |

---

### Question-Type Performance
This table reports results by **question type** (e.g., MCQ, comprehension, reasoning).

| Question Type           | gemma-2-2b-it | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Qwen2.5-3B-Instruct | granite-3.1-2b-instruct | FinanceParam |
|--------------------------|---------------|------------------------|------------------------|---------------------|--------------------------|--------------|
| Assertion or Reasoning   | 32.56         | 28.84                 | 35.35                 | 27.44              | 33.95                   | 29.77        |
| Fill in the blanks       | 35.66         | 27.97                 | 38.11                 | 44.06              | 33.92                   | 44.76        |
| MCQ                      | 30.40         | 26.29                 | 31.71                 | 33.20              | 31.31                   | 31.53        |
| Match the column         | 24.37         | 20.17                 | 32.77                 | 31.09              | 30.25                   | 22.69        |
| Reading Comprehension    | 30.59         | 25.88                 | 31.76                 | 28.24              | 31.76                   | 30.59        |
| Rearrange the sequence   | 24.29         | 23.59                 | 29.10                 | 28.39              | 22.88                   | 25.14        |

---