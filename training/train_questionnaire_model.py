"""
Train an XGBoost questionnaire-based autism screening model.

Dataset: UCI Autism Screening Adult Dataset (ARFF)
Purpose: Structured behavioral risk signal generator.
         Outputs a probability score (0-1) and SHAP feature importances.

The model is used as ONE modality in the multi-modal fusion pipeline.
It is NOT a diagnostic tool — it's a risk signal from structured screening questions.

Domain mapping (for child-care context):
  A1  → Social Smile            (Social)
  A2  → Eye Contact             (Social)
  A3  → Pointing / Gestures     (Communication)
  A4  → Shared Attention        (Social)
  A5  → Pretend Play            (Communication)
  A6  → Follows Gaze            (Social)
  A7  → Sensory Sensitivity     (Behavior)
  A8  → Motor Mannerisms        (Behavior)
  A9  → Responds to Name        (Communication)
  A10 → Social Reciprocity      (Social)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH = r"D:\Autism\AutismData\autism+screening+adult\Autism-Adult-Data.arff"
SAVE_DIR  = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MODEL_PATH    = os.path.join(SAVE_DIR, "questionnaire_xgb.pkl")
METADATA_PATH = os.path.join(SAVE_DIR, "questionnaire_metadata.json")

# ── Domain mapping for questions ───────────────────────────────────────
QUESTION_DOMAINS = {
    "A1_Score":  {"label": "Social Smile",          "domain": "Social"},
    "A2_Score":  {"label": "Eye Contact",            "domain": "Social"},
    "A3_Score":  {"label": "Pointing / Gestures",    "domain": "Communication"},
    "A4_Score":  {"label": "Shared Attention",        "domain": "Social"},
    "A5_Score":  {"label": "Pretend Play",            "domain": "Communication"},
    "A6_Score":  {"label": "Follows Gaze",            "domain": "Social"},
    "A7_Score":  {"label": "Sensory Sensitivity",     "domain": "Behavior"},
    "A8_Score":  {"label": "Motor Mannerisms",        "domain": "Behavior"},
    "A9_Score":  {"label": "Responds to Name",        "domain": "Communication"},
    "A10_Score": {"label": "Social Reciprocity",      "domain": "Social"},
}

QUESTION_COLS = [f"A{i}_Score" for i in range(1, 11)]
DEMO_COLS     = ["age", "gender", "jundice", "austim"]  # non-leaky demographics
TARGET        = "Class/ASD"

# Columns to DROP (leakage or irrelevant)
DROP_COLS = ["result", "age_desc", "relation", "contry_of_res",
             "ethnicity", "used_app_before"]


def load_and_prepare():
    """Load ARFF, decode bytes, clean, encode, return X/y DataFrames."""
    print("Loading UCI Autism Screening Adult Dataset...")
    data, meta = arff.loadarff(DATA_PATH)
    df = pd.DataFrame(data)

    # Decode byte strings from ARFF
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    print(f"  Raw shape: {df.shape}")
    print(f"  Class distribution:\n{df[TARGET].value_counts().to_string()}")

    # Drop leakage columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Handle missing values
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"].fillna(df["age"].median(), inplace=True)

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # Encode target: YES=1, NO=0
    df[TARGET] = (df[TARGET] == "YES").astype(int)

    # Encode question scores to int
    for col in QUESTION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Encode categorical demographics
    le_gender = LabelEncoder()
    df["gender"] = le_gender.fit_transform(df["gender"].fillna("unknown"))

    df["jundice"] = (df["jundice"] == "yes").astype(int)
    df["austim"]  = (df["austim"] == "yes").astype(int)

    # Feature engineering: domain aggregate scores
    df["social_sum"]        = df[["A1_Score","A2_Score","A4_Score","A6_Score","A10_Score"]].sum(axis=1)
    df["communication_sum"] = df[["A3_Score","A5_Score","A9_Score"]].sum(axis=1)
    df["behavior_sum"]      = df[["A7_Score","A8_Score"]].sum(axis=1)
    df["total_score"]       = df[QUESTION_COLS].sum(axis=1)

    feature_cols = (QUESTION_COLS + DEMO_COLS +
                    ["social_sum", "communication_sum", "behavior_sum", "total_score"])

    X = df[feature_cols]
    y = df[TARGET]

    print(f"  Final features: {X.shape[1]} columns")
    print(f"  Positive (ASD=1): {y.sum()} / {len(y)} ({100*y.mean():.1f}%)")

    return X, y, feature_cols


def train_model(X, y, feature_cols):
    """Train XGBoost, evaluate, save model + metadata."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

    # XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n  ✓ Test Accuracy : {acc:.4f}")
    print(f"  ✓ Test AUC-ROC  : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-ASD", "ASD"]))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"\n  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance (built-in)
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top Feature Importances:")
    for feat, imp in sorted_imp[:10]:
        bar = "█" * int(imp * 50)
        label = QUESTION_DOMAINS.get(feat, {}).get("label", feat)
        print(f"    {label:25s} ({feat:16s}): {imp:.4f} {bar}")

    # SHAP values
    print("\n  Computing SHAP values...")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = dict(zip(feature_cols, mean_shap.tolist()))
        print("  ✓ SHAP values computed")
    except Exception as e:
        print(f"  ⚠ SHAP failed: {e}")
        shap_importance = importance

    # Domain-level SHAP aggregation
    domain_shap = {"Social": 0.0, "Communication": 0.0, "Behavior": 0.0}
    for feat, val in shap_importance.items():
        domain = QUESTION_DOMAINS.get(feat, {}).get("domain")
        if domain:
            domain_shap[domain] += val
    total_domain = sum(domain_shap.values()) or 1.0
    domain_shap = {k: round(v / total_domain, 3) for k, v in domain_shap.items()}
    print(f"\n  Domain-Level Risk Profile: {domain_shap}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  ✓ Model saved: {MODEL_PATH}")

    # Save metadata
    metadata = {
        "model_type": "XGBClassifier",
        "dataset": "UCI Autism Screening Adult",
        "features": feature_cols,
        "question_domains": QUESTION_DOMAINS,
        "accuracy": round(acc, 4),
        "auc_roc": round(auc, 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "feature_importance": {k: round(v, 4) for k, v in sorted_imp},
        "shap_importance": {k: round(v, 4) for k, v in shap_importance.items()},
        "domain_profile": domain_shap,
        "note": "Proxy model trained on adult screening data. Adaptable to M-CHAT pediatric instruments.",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {METADATA_PATH}")

    return model, metadata


if __name__ == "__main__":
    X, y, feature_cols = load_and_prepare()
    model, metadata = train_model(X, y, feature_cols)

    # Demo: simulate a screening input
    print("\n" + "=" * 60)
    print("DEMO: Simulated Screening Input")
    print("=" * 60)
    sample = pd.DataFrame([{
        "A1_Score": 1, "A2_Score": 1, "A3_Score": 0, "A4_Score": 1,
        "A5_Score": 0, "A6_Score": 0, "A7_Score": 1, "A8_Score": 1,
        "A9_Score": 0, "A10_Score": 0,
        "age": 4, "gender": 1, "jundice": 0, "austim": 0,
        "social_sum": 3, "communication_sum": 0, "behavior_sum": 2, "total_score": 5,
    }])
    prob = model.predict_proba(sample)[0][1]
    print(f"  Questionnaire Risk Score: {prob:.4f}")
    print(f"  Interpretation: {'HIGH RISK' if prob > 0.6 else 'MODERATE' if prob > 0.3 else 'LOW RISK'}")
