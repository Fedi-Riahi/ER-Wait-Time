🏥 ER Wait Time Prediction — ML + FastAPI + Frontend
📌 Overview

This project is an end-to-end Machine Learning system with a FastAPI backend and a Vite frontend that predicts Emergency Room (ER) waiting time based on patient and hospital data.

It demonstrates the full ML workflow:
data processing → model training → evaluation → deployment → UI interaction

🎯 Objectives
Build a complete ML pipeline
Compare multiple regression models
Evaluate performance using standard metrics
Track experiments with MLflow
Expose predictions via a FastAPI backend
Provide a modern frontend UI for interaction
🧠 Models Used
Linear Regression
Ridge Regression
Support Vector Regression (SVR)
Random Forest
📊 Metrics
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
R² Score
📁 Project Structure
backend/        # FastAPI app
src/            # ML pipeline (training, preprocessing, etc.)
frontend/       # Vite + React frontend
models/         # Saved models (ignored in git)
data/           # Dataset (ignored in git)
mlruns/         # MLflow logs (ignored in git)
⚙️ Installation
Backend setup
uv sync
Frontend setup
cd frontend
pnpm install
▶️ Run the Project
1. Start Backend (FastAPI)
uv run uvicorn backend.main:app --reload --port 8000

👉 API Docs: http://127.0.0.1:8000/docs

2. Start Frontend (Vite)
cd frontend
pnpm dev

👉 Frontend runs on: http://localhost:5173

🧪 Train Models
uv run python src/train.py
📈 MLflow UI
mlflow ui

👉 Open: http://127.0.0.1:5000

🔗 Frontend–Backend Communication

The frontend communicates with the FastAPI backend via REST API:

Endpoint: /predict
Method: POST
Input: patient + hospital features
Output: predicted ER waiting time
📌 Notes
Synthetic data is generated if no dataset is provided
Large files (data/, models/, mlruns/) are ignored in Git
Frontend built using Vite + pnpm + React
