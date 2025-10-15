# Machine Learning–Assisted Optimization of Dietary Intervention Against Dementia Risk

This repository contains a full-stack, interpretable machine learning system that learns personalized dietary intervention rules to reduce dementia risk using causal inference and heterogeneous treatment effect estimation.

Core features:
- Target trial emulation, propensity scoring, IPW, and doubly robust estimation
- Heterogeneous treatment effect modeling (DRLearner, T-learner fallback)
- Policy learning for personalized diet recommendations under constraints
- Explainability (SHAP) and subgroup reporting
- Population-level policy rollout simulation
- Streamlit app for clinicians/patients and FastAPI endpoints
- MLflow-ready training pipeline

Quick start (Windows PowerShell):
1) Create and activate a virtual environment
   - py -3 -m venv .venv
   - .\.venv\Scripts\Activate.ps1
2) Install dependencies
   - pip install -r requirements.txt
3) Configure dataset path and columns in configs/config.yaml
4) Run training pipeline (first pass works with placeholder columns; update them for your data)
   - python scripts/train_pipeline.py
5) Launch the Streamlit app
   - streamlit run src/app/streamlit_app.py
6) Optional: Run the FastAPI server
   - uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Dataset path is initially set to your folder:
C:\\Users\\H P\\Downloads\\jovac presentation

You should update configs/config.yaml to match your actual column names (features, treatment, outcome, time columns, etc.).

Docker (optional):
- docker build -t dementia-diet-policy .
- docker run -p 8501:8501 -p 8000:8000 dementia-diet-policy

Structure:
- configs/ — configuration (dataset path, columns, modeling options)
- src/ — Python package for data, causal, models, policy, explainability, simulation, app, and api
- scripts/ — orchestration scripts
- models/ — saved artifacts

Notes:
- Some libraries (econml, scikit-survival) are optional; the code falls back gracefully when missing.
- For survival analysis, integrate lifelines/scikit-survival after confirming your outcome and censoring structure.
