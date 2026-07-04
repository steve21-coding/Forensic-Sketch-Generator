# Forensic Sketch Generator Pipeline

An enterprise-grade, decoupled forensic composite synthesis and facial biometric matching pipeline. The system accepts natural language witness descriptions, synthesizes high-fidelity suspect portraits using Stable Diffusion XL (SDXL), extracts semantic facial components using BiSeNet, and executes real-time vector similarity matching against a facial database utilizing a FAISS index.

## 1. Architectural Overview

The application is engineered using modular software design principles, strictly separating web network configurations, data contracts, and heavy machine learning inference states. 



* **State Isolation (Singleton Pattern):** Heavy model weights (SDXL, InsightFace, BiSeNet) are managed via a centralized configuration registry. Memory allocation occurs exactly once upon ASGI server boot via FastAPI lifespan context handlers, mitigating runtime race conditions and memory leaks during concurrent traffic.
* **Environment Agnostic Fallback (Defensive Design):** The services layer features an automated hardware verification mechanism (`torch.cuda.is_available()`). On standard development systems lacking a CUDA-accelerated GPU, the application transparently activates an intelligent Mock Engine to facilitate local end-to-end smoke tests without environment failures.
* **Global Error Interception:** Custom domain exceptions (e.g., `FaceNotFoundError`, `IndexNotBuiltError`) are isolated from core business logic. A global HTTP exception handler captures errors at the middleware layer, translating system failures into pristine, structured JSON response payloads mapped to explicit HTTP status codes ($400\text{ Bad Request}$, $503\text{ Service Unavailable}$).

---

## 2. Repository Structure

```text
Forensic Sketch Generator/
├── backend/
│   ├── config.py              # Singleton Model Registry & application lifecycle manager
│   ├── errors.py              # Custom domain exceptions & global error interceptors
│   ├── main.py                # FastAPI routing controller & application entry point
│   ├── schemas.py             # Pydantic data validation & type coercion models
│   └── services.py            # ML inference, semantic parsing, & FAISS vector operations
├── frontend/
│   └── index.html             # High-fidelity operator control dashboard
├── mock_database/             # Resolution-optimized synthetic suspect tracking images
│   ├── suspect_01.jpg
│   ├── suspect_02.jpg
│   └── suspect_03.jpg
├── notebooks/
│   └── colab_setup.ipynb      # Cloud-based GPU infrastructure provisioning notebook
├── .gitignore                 # Excludes raw model checkpoints, local datasets, and caches
└── README.md                  # Comprehensive technical documentation
