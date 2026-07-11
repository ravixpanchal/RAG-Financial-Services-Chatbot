# RAG-Financial-Services-Chatbot

# 🤖 RAG Chatbot

<p align="center">
  <img src="docs/banner.png" alt="RAG Chatbot Banner" width="100%">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![FAISS](https://img.shields.io/badge/Vector_Database-FAISS-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</p>

<p align="center">

**A Complete Retrieval-Augmented Generation (RAG) Ecosystem consisting of five AI-powered applications for intelligent document retrieval, finance analysis, GST report generation, MSME report generation, and API experimentation.**

</p>

---

# 📖 Overview

The **RAG Chatbot** is a comprehensive AI platform developed during our internship at **BISAG-N**.

Unlike a traditional chatbot, this repository contains **five integrated AI modules**, each solving a unique real-world problem while sharing the same Retrieval-Augmented Generation (RAG) pipeline.

The project demonstrates practical implementation of

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector Databases
- Prompt Engineering
- AI-powered Report Generation
- Semantic Search
- Document Question Answering

---

# 🚀 Live Demo

| Project | Deployment |
|----------|------------|
| 💰 FinanceParam Model | https://example.com |
| ⚡ Query Optimizer | https://example.com |
| 🤖 SARVAM API Tester | https://example.com |
| 📊 GST Report Generator | https://example.com |
| 📈 MSME Report Generator | https://example.com |

---

# 📂 Project Modules

| No | Project | Description | Folder |
|----|----------|-------------|--------|
| 1 | FinanceParam Model (Quantize & Unquantize) | Quantization experiments using QLoRA and comparison with original FinanceParam model | [FinanceParam-Quantization](FinanceParam-Quantization/) |
| 2 | Query Optimizer | Optimizes user prompts before sending them to the LLM | [Query-Optimizer](Query-Optimizer/) |
| 3 | SARVAM API Tester | Interactive interface to test and evaluate SARVAM API | [SARVAM-API-Tester](SARVAM-API-Tester/) |
| 4 | GST Report Generator | Generates AI-powered GST reports from Excel files using RAG | [GST-Report-Generator](GST-Report-Generator/) |
| 5 | MSME Report Generator | Generates MSME reports and business insights | [MSME-Report-Generator](MSME-Report-Generator/) |

---

# 🎥 Demo Video

Watch the complete project demonstration.

▶ https://youtube.com/example

---

# 🖼 Project Preview

## Home Page

<p align="center">
<img src="docs/screenshots/home.png" width="900">
</p>

---

## FinanceParam Model

<p align="center">
<img src="docs/screenshots/financeparam.gif" width="900">
</p>

---

## Query Optimizer

<p align="center">
<img src="docs/screenshots/query.gif" width="900">
</p>

---

## SARVAM API Tester

<p align="center">
<img src="docs/screenshots/sarvam.gif" width="900">
</p>

---

## GST Report Generator

<p align="center">
<img src="docs/screenshots/gst.gif" width="900">
</p>

---

## MSME Report Generator

<p align="center">
<img src="docs/screenshots/msme.gif" width="900">
</p>

---

# 🏗 Repository Structure

```
RAG-Chatbot
│
├── README.md
│
├── docs
│   ├── banner.png
│   ├── architecture.png
│   ├── report.pdf
│   └── screenshots
│       ├── home.png
│       ├── financeparam.gif
│       ├── query.gif
│       ├── sarvam.gif
│       ├── gst.gif
│       └── msme.gif
│
├── FinanceParam-Quantization
│   ├── Quantization.ipynb
│   ├── UnQuantization.ipynb
│   └── README.md
│
├── Query-Optimizer
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── SARVAM-API-Tester
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── GST-Report-Generator
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── MSME-Report-Generator
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
└── LICENSE
```

---

# 🏛 System Architecture

<p align="center">
<img src="docs/architecture.png" width="900">
</p>

---

## Architecture Workflow

```
                    User
                      │
                      ▼
              Streamlit Web App
                      │
      ┌───────────────┴──────────────┐
      │                              │
      ▼                              ▼
Query Optimizer               Document Upload
      │                              │
      └───────────────┬──────────────┘
                      ▼
             Embedding Generation
                      │
                      ▼
                 Vector Database
                    (FAISS)
                      │
                      ▼
             Relevant Documents
                      │
                      ▼
          FinanceParam / SARVAM API
                      │
                      ▼
              AI Generated Response
                      │
                      ▼
      Report Generation / Chat Interface
```

---

# ⭐ Key Features

- Retrieval-Augmented Generation (RAG)
- Finance Domain Chatbot
- AI-powered Report Generation
- Semantic Search
- Vector Database Retrieval
- Query Optimization
- Prompt Engineering
- Streamlit Web Applications
- Excel Report Analysis
- MSME Report Generation
- GST Analytics
- SARVAM API Integration
- Modular Architecture
- Easy Deployment

---

# 💻 Technology Stack

## Programming Languages

- Python
- HTML
- CSS
- JavaScript

---

## Artificial Intelligence

- LangChain
- HuggingFace Transformers
- FinanceParam
- Sentence Transformers
- PEFT
- QLoRA
- PyTorch

---

## Vector Database

- FAISS

---

## Frameworks

- Streamlit
- FastAPI

---

## Libraries

- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- OpenPyXL

---

# 📊 Applications

The RAG Chatbot can be deployed in

- Banking
- Finance
- GST Consultation
- MSME Analysis
- Government Organizations
- Enterprise Knowledge Base
- Educational Institutions
- Legal Document Search
- Customer Support
- AI Assistants
- Business Intelligence
- Research Organizations

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/USERNAME/RAG-Chatbot.git
```

---

## Move into Repository

```bash
cd RAG-Chatbot
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Any Project

Example

```bash
cd GST-Report-Generator

streamlit run app.py
```

---

# 📌 Individual Projects

Every module contains

- README
- Source Code
- Screenshots
- Requirements
- Deployment Guide

---

# 🔮 Future Scope

- Voice Assistant
- OCR Integration
- Multi-language Support
- Docker Deployment
- Kubernetes Deployment
- Cloud Integration
- Authentication
- User Dashboard
- Admin Dashboard
- Report Export
- Mobile Application

---

# 📚 Research & Learning

This repository demonstrates concepts including

- Retrieval-Augmented Generation
- Prompt Engineering
- Semantic Search
- Vector Embeddings
- Quantization
- Large Language Models
- Information Retrieval
- AI-powered Report Generation

---

# 👨‍💻 Contributors

| Name | Role |
|------|------|
| Ravi Panchal | AI Developer |
| Arup Das | AI Developer |
| Anurag Jaiswal | AI Developer |

---

# 🏢 Organization

Developed during the **Summer Internship at BISAG-N (Bhaskaracharya National Institute for Space Applications and Geo-informatics).**

---

# 🙏 Acknowledgements

We sincerely thank **BISAG-N**, our mentors, and the entire development team for their continuous guidance and support throughout the internship.

---

# 📜 License

Licensed under the MIT License.

See the **LICENSE** file for details.

---

# ⭐ Support

If you found this repository useful,

⭐ Star this repository

🍴 Fork it

🛠 Contribute to improve it

📢 Share it with others

---

<p align="center">

## 🚀 Building Intelligent AI Solutions with Retrieval-Augmented Generation

Made with ❤️ by the RAG Chatbot Team

</p>
