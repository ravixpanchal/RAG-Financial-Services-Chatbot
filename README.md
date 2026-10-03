# 📈 BharatGenFinance: Financial AI Analyst & RAG Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FinanceParam-FFD21E.svg)](https://huggingface.co/bharatgenai/FinanceParam)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-0055FF.svg)](https://github.com/facebookresearch/faiss)

A high-performance financial analysis and document RAG system powered by the **`bharatgenai/FinanceParam`** LLM. Built for local developer machines (such as 4GB VRAM GPUs) by offloading heavy 7B model inference to **Google Colab GPU** (T4/V100/A100) and persisting model weights & FAISS vector stores permanently on **Google Drive**.

---

## 🏗️ Architecture Overview

```
                         YOUR LAPTOP (VS Code)
┌─────────────────────────────────────────────────────────────────┐
│ Streamlit Frontend (app_streamlit.py)                           │
│  - Interactive Financial Chat UI with chat memory               │
│  - Document Uploader (PDF / TXT / CSV)                          │
│  - Institutional Equity Analysis preset (prompt.txt)             │
│  - Real-time Backend Health & VRAM Monitoring                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 │ HTTPS Tunnel (Cloudflare / Ngrok)
                                 ▼
                         GOOGLE COLAB (GPU)
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI Backend (colab_backend.py / Colab Notebook)             │
│  ├── /health    - System status & GPU VRAM monitor              │
│  ├── /chat      - RAG + FinanceParam model generation           │
│  ├── /upload    - PDF/TXT/CSV parsing, chunking, FAISS index    │
│  ├── /documents - View vector store metadata & chunk count      │
│  └── /clear     - Reset vector index                            │
│                                                                 │
│ Model: bharatgenai/FinanceParam (torch.bfloat16 / float16)      │
│ Engine: PyTorch safe_forward KV-Cache Interceptor (3-5s response)│
│ Vector Store: FAISS (sentence-transformers/all-MiniLM-L6-v2)    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                         GOOGLE DRIVE
┌─────────────────────────────────────────────────────────────────┐
│ Permanent Storage (/content/drive/MyDrive/BharatGenFinance/)    │
│  ├── FinanceParam Model Weights Cache                           │
│  └── FAISS Vector DB Index (faiss_index/)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 Key Features

- **🚀 Offloaded Colab GPU Inference**: Runs 7B parameter financial LLM on Google Colab GPU (T4/V100), bypassing local hardware VRAM limitations (e.g., RTX 2050 4GB).
- **⚡ Fast Response Speed (3–5 seconds)**: PyTorch `safe_forward` model interceptor enables KV-caching safely under the hood while maintaining 100% compatibility with HuggingFace `transformers` 4.45+.
- **📁 Persistent Google Drive Storage**: Model weights and FAISS vector indices are stored in `/content/drive/MyDrive/BharatGenFinance/` so restarts are instantaneous without re-downloading model files.
- **📄 RAG Document Indexing**: Upload financial statements, annual reports, earnings transcripts, or equity research notes in PDF, TXT, or CSV formats.
- **📈 Institutional Equity Analysis Preset**: Integrates [`prompt.txt`](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/BharatGenFinance/prompt.txt) for structured equity research covering Revenue Drivers, Profitability, Balance Sheet Health, Peer Comparison, Valuation Multiples, and Rating Recommendations.
- **🌐 Dual Tunneling Support**: Automatically uses **Cloudflare Tunnels** (`pycloudflared`) out-of-the-box (no registration needed) or **Ngrok** if an authtoken is provided.

---

## 📁 Repository Structure

```
BharatGenFinance/
├── app_streamlit.py                  # Local Streamlit UI (VS Code)
├── colab_backend.py                  # FastAPI GPU Backend Script
├── BharatGenFinance_Colab_Server.ipynb # 1-Click Google Colab Server Notebook
├── prompt.txt                        # Institutional Equity Analysis Prompt
├── requirements_laptop.txt           # Laptop dependencies (Streamlit, Requests, Pandas)
├── requirements_colab.txt            # Google Colab GPU backend dependencies
├── test_doc.txt                      # Sample testing document
└── README.md                         # Project Documentation
```

---

## 🚀 Quick Start Guide

### Step 1: Launch Backend on Google Colab

1. Open **Google Colab** ([colab.research.google.com](https://colab.research.google.com)).
2. Select **Runtime -> Change runtime type** and set hardware accelerator to **T4 GPU**.
3. Upload and open [`BharatGenFinance_Colab_Server.ipynb`](file:///home/ravi/Desktop/Study%20Material/Projects/Personal/BharatGenFinance/BharatGenFinance_Colab_Server.ipynb).
4. Click **Runtime -> Run all** (or execute cells in order):
   - **Step 1**: Mount Google Drive (`drive.mount('/content/drive')`).
   - **Step 2**: Install backend dependencies.
   - **Step 3**: Write backend script (`colab_backend.py`).
   - **Step 4**: Start FastAPI server & public tunnel.
5. Copy the printed **Public Backend URL**:
   ```text
   =================================================================
   👉 PUBLIC BACKEND URL FOR VS CODE: https://xxxx.trycloudflare.com
   👉 Copy and paste this URL into your Streamlit UI sidebar!
   =================================================================
   ```

---

### Step 2: Launch Streamlit UI on Laptop (VS Code)

1. Open a terminal in the project root:
   ```bash
   source venv/bin/activate
   pip install -r requirements_laptop.txt
   ```
2. Launch the Streamlit interface:
   ```bash
   streamlit run app_streamlit.py
   ```
3. In the Streamlit sidebar:
   - Paste your **Public Backend URL** (e.g. `https://xxxx.trycloudflare.com` or Ngrok URL).
   - Click **🔌 Check Connection** to verify green status badge and VRAM readout.

---

## 🛠️ API Endpoints Reference

| Endpoint | Method | Payload / Params | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | System status, VRAM usage, and indexed document chunk count |
| `/chat` | `POST` | `ChatRequest` JSON | RAG context retrieval + FinanceParam model response generation |
| `/upload` | `POST` | `multipart/form-data` | Multi-file document upload (PDF/TXT/CSV), chunking, and FAISS indexing |
| `/documents`| `GET` | None | Total chunk count and FAISS storage directory path |
| `/clear` | `POST` | None | Reset vector store index and remove cached vector files |

### Example Chat Request (`POST /chat`)

```json
{
  "message": "What is the capital adequacy ratio requirement by RBI?",
  "history": [],
  "mode": "RAG Document Q&A",
  "system_prompt": "You are an expert AI financial analyst.",
  "temperature": 0.7,
  "max_new_tokens": 512,
  "top_k": 4
}
```

---

## 🔧 Engineering & Reliability Highlights

- **PyTorch Interceptor for KV Caching**: Resolved `DynamicCache index 0 out of range` and HuggingFace 4.45+ `Invalid cache_implementation (legacy)` errors by wrapping `model.forward` with a custom interceptor (`safe_forward`) that converts `DynamicCache` instances to legacy key/value tuples dynamically.
- **Asyncio Event Loop Conflict Prevention**: Runs the Uvicorn FastAPI server in a dedicated background daemon thread (`threading.Thread(target=start_uvicorn, daemon=True)`) to eliminate event loop collisions in Google Colab.
- **Token Bounds Clamping**: Sanitizes pad token IDs and clamps `input_ids` to `[0, vocab_size - 1]` to eliminate PyTorch CUDA device-side assertion failures.
- **Cloudflare 100s Timeout Defense**: High-speed KV-caching keeps response generation under 5 seconds, preventing Cloudflare Error 524 timeouts.
