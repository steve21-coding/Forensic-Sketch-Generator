# 🔴 Forensic Sketch Generator — Automated Composite Synthesis & Matching System

The Forensic Sketch Generator is an end-to-end forensic pipeline designed to bridge the gap between human memory and criminal databases. The system accepts a casual natural language description from a witness, synthesizes a high-fidelity suspect composite using generative AI, and immediately executes a hybrid biometric and semantic search against a database of mugshots to rank potential suspects in seconds.

## 🚀 Key Features
* **AI Composite Generation:** Leverages **Stable Diffusion XL (Juggernaut-XL-v9)** to synthesize realistic, front-facing police-style portraits from witness prompts.
* **Semantic Facial Parsing:** Uses **BiSeNet** to extract exact facial traits like skin tone (HSV), hair color (RGB), and facial hair presence.
* **Biometric Re-ranking:** Deploys **InsightFace** for deep face embeddings combined with a **FAISS** vector index to calculate final similarity scores.
* **Interactive Dashboard:** A sleek, responsive dark/light mode UI designed for forensic operators to manage the entire generation and search flow.

## 📁 Directory Structure
```text
Forensic Sketch Generator/
├── backend/
│   └── fastapi_backend.py     # FastAPI REST API wrapper for the ML models
├── frontend/
│   └── index.html             # Interactive dashboard frontend
├── notebooks/
│   └── colab_setup.ipynb      # Google Colab environment orchestration script
├── .gitignore                 # Files excluded from version control
└── README.md                  # Project documentation
