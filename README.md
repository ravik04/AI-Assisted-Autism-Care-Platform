# AI-Assisted Autism Care Platform

> Multi-modal AI screening prototype for early autism detection — **TELIPORT Season 3 / Tata Elxsi**

## Overview

An end-to-end, neurodiversity-affirming platform that combines **9 trained ML models** across **7 modalities** with **4 intelligent AI agents** to provide multi-modal autism screening, clinical assessment, therapy recommendations, and longitudinal monitoring.

### Key Features

- **7 Modalities**: Facial analysis, behavior video, questionnaire, eye-tracking, pose/skeleton, audio/speech, EEG/neural
- **9 ML Models**: CNN, LSTM, XGBoost, Ridge, 1D-CNN, GRU, Cross-Modal Attention Fusion
- **4 LLM-Powered Agents**: Screening, Clinical (DSM-5), Therapy (RAG), Monitoring (EWMA/CUSUM)
- **Attention-Based Fusion**: Cross-modal attention with missing-modality support
- **GDPR/COPPA Consent Management**: Category-level consent with audit trail
- **RLHF Feedback Loop**: Clinician feedback collection for continuous improvement
- **Explainability Engine**: Per-modality explanations, feature importance, neurodiversity-affirming language
- **Role-Based Views**: Clinician, Parent, Therapist, Researcher perspectives
- **Child Profiles & Longitudinal Tracking**: Session history and progress monitoring

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)                   │
│  Dashboard │ Screening │ Profile │ Therapy │ Progress │ Reports │
│  + Consent Modal │ Role Selector │ Audio/EEG Upload        │
│  + RLHF Feedback │ Explainability │ Neurodiversity Banner  │
└────────────────────────┬───────────────────────────────────┘
                         │ REST API (~15 endpoints)
┌────────────────────────┴───────────────────────────────────┐
│                  FastAPI Backend v5 (Port 8000)             │
│                                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │   Face   │ │ Behavior │ │  Quest.  │ │Eye-Track │     │
│  │CNN 82.5% │ │LSTM 61.5%│ │XGB 100%  │ │XGB 67.5% │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │  Pose    │ │  CARS    │ │  Audio   │ │   EEG    │     │
│  │XGB 96.7% │ │Ridge     │ │CNN+GRU   │ │ 1D-CNN   │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────────────────────────────────┐                  │
│  │   Attention Fusion (89.5% / 0.95)   │                  │
│  └──────────────────────────────────────┘                  │
│                                                            │
│  ┌─────────────────── AI Agents ──────────────────────┐   │
│  │ Screening: Bayesian + Cross-Modal Attention        │   │
│  │ Clinical:  LLM-Powered (GPT-4o-mini) + DSM-5      │   │
│  │ Therapy:   RAG over ABA/ESDM/PECS protocols        │   │
│  │ Monitoring: EWMA + Forecasting + CUSUM Detection   │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─────────────────── Modules ────────────────────────┐   │
│  │ Consent (GDPR/COPPA) │ RLHF Feedback │ Storage     │   │
│  │ Explainability │ Audio Features │ EEG Features      │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

## Models

| Model | Type | Accuracy | AUC | Modality |
|-------|------|----------|-----|----------|
| Face Classifier | MobileNetV2 CNN | 82.5% | — | Facial |
| Behavior LSTM | MobileNetV2 + LSTM | 61.5% | — | Behavior |
| Questionnaire | XGBoost | 100% | 100% | Questionnaire |
| Eye-Tracking | XGBoost | 67.5% | 74.5% | Eye-Tracking |
| Pose/Skeleton | XGBoost | 96.7% | 99.3% | Pose |
| CARS Severity | Ridge Regression | MAE 4.85 | — | Clinical |
| Audio/Speech | CNN + GRU | 50.0%* | — | Audio |
| EEG Neural | 1D-CNN | 98.0% | 100% | EEG |
| Attention Fusion | Cross-Modal Attention | 89.5% | 95.0% | Multi-modal |

\* Audio model trained on synthetic data — accuracy expected to improve with real clinical recordings.

## AI Agents (LLM-Powered)

- **Screening Agent** — Reliability-weighted ensemble fusion across 7 modalities, Bayesian confidence intervals (Beta posterior), cross-modal attention scoring, adaptive thresholds
- **Clinical Agent** — GPT-4o-mini with DSM-5 criteria & M-CHAT-R/F guidelines; generates structured clinical notes with severity estimates and DSM-5 indicator mapping
- **Therapy Agent** — RAG engine with TF-IDF retrieval over 27 evidence-based techniques across 5 protocols (ABA, ESDM, PECS, Sensory Integration, Social Skills); includes audio/EEG modality mappings
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

# 5. Train models (or use pre-trained models in saved_models/)
python training/train_face_classifier.py
python training/train_behavior_lstm.py
python training/train_questionnaire_model.py
python training/train_eye_tracking_model.py
python training/train_pose_model.py
python training/train_eye_tracking_cars.py
python training/train_audio_model.py
python training/train_eeg_model.py
python training/train_fusion_model.py

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
| GET | `/api/status` | Model & agent status (9 models) |
| GET | `/api/model-info` | Model metrics & metadata |
| POST | `/api/analyze` | Upload image/video for analysis |
| POST | `/api/questionnaire` | Submit screening questionnaire |
| POST | `/api/fuse` | Multi-modal score fusion (attention-based) |
| GET | `/api/history` | Session history |
| POST | `/api/clear` | Clear session data |
| POST | `/api/analyze-audio` | Audio/speech analysis (file or synthetic) |
| POST | `/api/analyze-eeg` | EEG neural analysis (CSV or synthetic) |
| POST | `/api/consent` | Grant GDPR/COPPA consent |
| GET | `/api/consent/{child_id}` | Get consent summary |
| DELETE | `/api/consent/{child_id}` | Revoke consent |
| POST | `/api/feedback` | Submit RLHF feedback |
| GET | `/api/feedback/summary` | Aggregated feedback stats |
| POST | `/api/children` | Create child profile |
| GET | `/api/children` | List all children |
| GET | `/api/children/{child_id}` | Profile + sessions + longitudinal |
| POST | `/api/explain` | Full explainability report |

## Project Structure

```
├── agents/
│   ├── screening_agent.py      # Bayesian fusion (7 modalities)
│   ├── clinical_agent.py       # LLM-powered (GPT-4o-mini)
│   ├── therapy_agent.py        # RAG recommendation engine
│   ├── monitoring_agent.py     # Time-series forecasting
│   └── therapy_knowledge_base.json  # ABA/ESDM/PECS + audio/EEG
├── backend/
│   └── main.py                 # FastAPI server v5 (~15 endpoints)
├── frontend/
│   ├── index.html              # Dashboard UI + consent modal
│   ├── style.css               # Styling + consent/feedback/explainability
│   └── app.js                  # Frontend logic + audio/EEG/RLHF handlers
├── models/
│   ├── attention_fusion.py     # Cross-modal attention fusion model
│   └── temporal_transformer.py # Temporal Transformer architecture
├── training/
│   ├── train_face_classifier.py
│   ├── train_behavior_lstm.py
│   ├── train_questionnaire_model.py
│   ├── train_eye_tracking_model.py
│   ├── train_pose_model.py
│   ├── train_eye_tracking_cars.py
│   ├── train_audio_model.py    # Audio CNN+GRU training
│   ├── train_eeg_model.py      # EEG 1D-CNN training
│   └── train_fusion_model.py   # Attention fusion training
├── saved_models/               # 9 trained model files
├── utils/
│   ├── llm_client.py           # OpenAI GPT wrapper
│   ├── gradcam.py              # Grad-CAM visualization
│   ├── logger.py               # Result logging
│   ├── audio_features.py       # 40-feature audio extraction (librosa)
│   ├── eeg_features.py         # 20-feature EEG extraction (scipy)
│   ├── consent.py              # GDPR/COPPA consent management
│   ├── feedback.py             # RLHF feedback collection
│   ├── storage.py              # JSON persistence (children/sessions)
│   └── explainability.py       # Per-modality explanations + feature importance
├── data/                       # Runtime data (consent, feedback, sessions)
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── README.md
```

## Tech Stack

- **ML**: TensorFlow 2.x / Keras 3, XGBoost, scikit-learn, OpenCV
- **Audio**: librosa, soundfile (MFCC, prosody, vocalization features)
- **EEG**: SciPy (band powers, asymmetry, complexity, connectivity)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **LLM**: OpenAI GPT-4o-mini (optional, with fallback)
- **Data**: NumPy, SciPy, Pillow

## Neurodiversity Statement

This platform uses neurodiversity-affirming language throughout. Autism is recognized as a natural variation in neurodevelopment — not a disease to be cured. All screening results describe behavioral patterns and support needs, not deficits.

## License

MIT