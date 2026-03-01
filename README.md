# AI-Assisted Autism Care Platform

> Multi-modal AI screening prototype for early autism detection — **TELIPORT Season 3 / Tata Elxsi**

## Overview

An end-to-end platform that combines **6 trained ML models** with **4 intelligent AI agents** to provide multi-modal autism screening, clinical assessment, therapy recommendations, and longitudinal monitoring.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)                   │
│  Dashboard │ Screening │ Profile │ Therapy │ Progress │ Reports │
└────────────────────────┬───────────────────────────────────┘
                         │ REST API
┌────────────────────────┴───────────────────────────────────┐
│                  FastAPI Backend (Port 8000)                │
│                                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │   Face   │ │ Behavior │ │  Quest.  │ │Eye-Track │     │
│  │CNN 82.5% │ │LSTM 61.5%│ │XGB 100%  │ │XGB 67.5% │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐                                │
│  │  Pose    │ │  CARS    │   Weighted Fusion Engine        │
│  │XGB 96.7% │ │Ridge     │   → Bayesian Confidence         │
│  └──────────┘ └──────────┘                                │
│                                                            │
│  ┌─────────────────── AI Agents ──────────────────────┐   │
│  │ Screening: Bayesian + Cross-Modal Attention        │   │
│  │ Clinical:  LLM-Powered (GPT-4o-mini) + DSM-5      │   │
│  │ Therapy:   RAG over ABA/ESDM/PECS protocols        │   │
│  │ Monitoring: EWMA + Forecasting + CUSUM Detection   │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

## Models

| Model | Type | Accuracy | AUC |
|-------|------|----------|-----|
| Face Classifier | MobileNetV2 CNN | 82.5% | — |
| Behavior LSTM | MobileNetV2 + LSTM | 61.5% | — |
| Questionnaire | XGBoost | 100% | 100% |
| Eye-Tracking | XGBoost | 67.5% | 74.5% |
| Pose/Skeleton | XGBoost | 96.7% | 99.3% |
| CARS Severity | Ridge Regression | MAE 4.85 | — |

## AI Agents (Stage 2 — LLM-Powered)

- **Screening Agent** — Reliability-weighted ensemble fusion, Bayesian confidence intervals (Beta posterior), cross-modal attention scoring, adaptive thresholds
- **Clinical Agent** — GPT-4o-mini with DSM-5 criteria & M-CHAT-R/F guidelines; generates structured clinical notes with severity estimates and DSM-5 indicator mapping
- **Therapy Agent** — RAG engine with TF-IDF retrieval over 27 evidence-based techniques across 5 protocols (ABA, ESDM, PECS, Sensory Integration, Social Skills); LLM-personalized plans
- **Monitoring Agent** — EWMA smoothing, linear regression forecasting (3-session horizon), CUSUM change-point detection, z-score anomaly flagging, velocity/acceleration metrics

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ravik04/AI-Assisted-Autism-Care-Platform.git
cd AI-Assisted-Autism-Care-Platform

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (optional — agents work without it)

# 5. Train models (or place pre-trained models in saved_models/)
python training/train_face_classifier.py
python training/train_behavior_lstm.py
python training/train_questionnaire_model.py
python training/train_eye_tracking_model.py
python training/train_pose_model.py
python training/train_eye_tracking_cars.py

# 6. Start backend
cd backend
python main.py
# API at http://localhost:8000 — docs at http://localhost:8000/docs

# 7. Start frontend (new terminal)
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Model & agent status |
| GET | `/api/model-info` | Model metrics & metadata |
| POST | `/api/analyze` | Upload image/video for analysis |
| POST | `/api/questionnaire` | Submit screening questionnaire |
| POST | `/api/fuse` | Manual multi-modal score fusion |
| GET | `/api/history` | Session history |
| POST | `/api/clear` | Clear session data |

## Project Structure

```
├── agents/
│   ├── screening_agent.py      # Bayesian + cross-modal attention
│   ├── clinical_agent.py       # LLM-powered (GPT-4o-mini)
│   ├── therapy_agent.py        # RAG recommendation engine
│   ├── monitoring_agent.py     # Time-series forecasting
│   └── therapy_knowledge_base.json  # ABA/ESDM/PECS protocols
├── backend/
│   └── main.py                 # FastAPI server (all endpoints)
├── frontend/
│   ├── index.html              # Dashboard UI
│   ├── style.css               # Styling
│   └── app.js                  # Frontend logic
├── training/                   # Model training scripts
├── saved_models/               # Trained model files (not in git)
├── utils/
│   ├── llm_client.py           # OpenAI GPT wrapper
│   ├── gradcam.py              # Grad-CAM visualization
│   └── logger.py               # Result logging
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── README.md
```

## Tech Stack

- **ML**: TensorFlow 2.x, XGBoost, scikit-learn, OpenCV
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **LLM**: OpenAI GPT-4o-mini (optional, with fallback)
- **Data**: NumPy, SciPy, Pillow

## License

MIT