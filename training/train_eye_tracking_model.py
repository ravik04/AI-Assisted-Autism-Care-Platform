"""
Train an eye-tracking gaze pattern classifier for autism screening.

Datasets used:
  1. Saliency4ASD (300 images × ASD scanpath + TD scanpath)
     - Fixation coordinates (x, y, duration) per image
     - Per-image feature extraction → aggregated per-group samples
  2. Eye-Tracking Dataset (25 ASD participant CSVs — enrichment)

Features extracted per scanpath:
  - Fixation count, duration stats (mean, std, max, total)
  - Saccade amplitude/velocity (distance between fixations)
  - Gaze dispersion (spatial spread)
  - Center bias (distance from screen center)
  - Scanpath length, convex hull area

Model: XGBoost → outputs probability score (0-1)
Purpose: Gaze-based risk signal for multi-modal fusion.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
SALIENCY_DIR = r"D:\Autism\AutismData\13960426\TrainingData\TrainingData"
ASD_SCAN_DIR = os.path.join(SALIENCY_DIR, "ASD")
TD_SCAN_DIR  = os.path.join(SALIENCY_DIR, "TD")

SAVE_DIR      = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MODEL_PATH    = os.path.join(SAVE_DIR, "eye_tracking_xgb.pkl")
SCALER_PATH   = os.path.join(SAVE_DIR, "eye_tracking_scaler.pkl")
METADATA_PATH = os.path.join(SAVE_DIR, "eye_tracking_metadata.json")

# Screen resolution from README
SCREEN_W, SCREEN_H = 1280, 1024

# ── Feature extraction from scanpath txt files ─────────────────────────

def parse_scanpath(filepath):
    """Parse a Saliency4ASD scanpath file.
    Format: Idx, x, y, duration (multiple subjects per image, separated by Idx=0 restarts)
    Returns list of subject scanpaths, each = list of (x, y, duration) tuples.
    """
    scanpaths = []
    current = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Idx"):
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    dur = float(parts[3])
                except ValueError:
                    continue

                if idx == 0 and len(current) > 0:
                    scanpaths.append(current)
                    current = []
                current.append((x, y, dur))

        if current:
            scanpaths.append(current)
    except Exception:
        pass
    return scanpaths


def extract_scanpath_features(fixations):
    """Extract features from a single scanpath (list of (x,y,dur) tuples)."""
    if len(fixations) < 2:
        return None

    xs = np.array([f[0] for f in fixations])
    ys = np.array([f[1] for f in fixations])
    durs = np.array([f[2] for f in fixations])

    feats = {}

    # ── Fixation count and duration ──
    feats["n_fixations"] = len(fixations)
    feats["dur_mean"] = durs.mean()
    feats["dur_std"]  = durs.std()
    feats["dur_max"]  = durs.max()
    feats["dur_min"]  = durs.min()
    feats["dur_total"] = durs.sum()
    feats["dur_median"] = np.median(durs)

    # ── Spatial features ──
    feats["x_mean"] = xs.mean()
    feats["x_std"]  = xs.std()
    feats["y_mean"] = ys.mean()
    feats["y_std"]  = ys.std()

    # Gaze dispersion
    feats["gaze_dispersion"] = np.sqrt(xs.var() + ys.var())

    # Center bias (how far from screen center)
    cx, cy = SCREEN_W / 2, SCREEN_H / 2
    dist_center = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    feats["center_dist_mean"] = dist_center.mean()
    feats["center_dist_std"]  = dist_center.std()

    # ── Saccade features (inter-fixation) ──
    dx = np.diff(xs)
    dy = np.diff(ys)
    saccade_amp = np.sqrt(dx**2 + dy**2)

    feats["saccade_amp_mean"] = saccade_amp.mean()
    feats["saccade_amp_std"]  = saccade_amp.std()
    feats["saccade_amp_max"]  = saccade_amp.max()

    # Saccade velocity (amplitude / inter-fixation duration)
    dur_pairs = (durs[:-1] + durs[1:]) / 2
    dur_pairs = np.maximum(dur_pairs, 1)  # avoid div by 0
    saccade_vel = saccade_amp / dur_pairs
    feats["saccade_vel_mean"] = saccade_vel.mean()
    feats["saccade_vel_std"]  = saccade_vel.std()

    # ── Scanpath metrics ──
    feats["scanpath_length"] = saccade_amp.sum()
    feats["scanpath_duration"] = durs.sum()

    # Scanpath regularity (ratio of cumulative length to direct distance)
    direct_dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    feats["scanpath_efficiency"] = direct_dist / max(feats["scanpath_length"], 1)

    # Convex hull area (approximate spatial coverage)
    try:
        from scipy.spatial import ConvexHull
        if len(set(zip(xs, ys))) >= 3:
            hull = ConvexHull(np.column_stack([xs, ys]))
            feats["convex_hull_area"] = hull.volume  # 2D: volume = area
        else:
            feats["convex_hull_area"] = 0
    except Exception:
        feats["convex_hull_area"] = 0

    # ── Duration distribution entropy ──
    dur_hist, _ = np.histogram(durs, bins=10, density=True)
    dur_hist = dur_hist[dur_hist > 0]
    feats["dur_entropy"] = -np.sum(dur_hist * np.log2(dur_hist + 1e-10))

    # ── Revisitation (how often gaze returns to previously visited areas) ──
    grid_cells = set()
    revisits = 0
    for x, y in zip(xs, ys):
        cell = (int(x // 50), int(y // 50))
        if cell in grid_cells:
            revisits += 1
        grid_cells.add(cell)
    feats["revisitation_ratio"] = revisits / max(len(fixations), 1)

    return feats


def build_dataset():
    """Build feature matrix from Saliency4ASD scanpath files."""
    all_samples = []

    for label, scan_dir, class_val in [("ASD", ASD_SCAN_DIR, 1), ("TD", TD_SCAN_DIR, 0)]:
        files = sorted([f for f in os.listdir(scan_dir) if f.endswith(".txt")])
        print(f"  {label}: {len(files)} scanpath files")

        for fname in files:
            filepath = os.path.join(scan_dir, fname)
            scanpaths = parse_scanpath(filepath)

            # Each file may contain multiple subjects' scanpaths for one image
            for sp_idx, sp in enumerate(scanpaths):
                feats = extract_scanpath_features(sp)
                if feats is not None:
                    feats["class"] = class_val
                    feats["image_id"] = fname
                    feats["subject_idx"] = sp_idx
                    all_samples.append(feats)

    df = pd.DataFrame(all_samples)
    print(f"\n  Total samples: {len(df)}")
    print(f"  ASD: {(df['class']==1).sum()}, TD: {(df['class']==0).sum()}")
    return df


def train_model(df):
    """Train XGBoost on extracted gaze features."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    drop = ["class", "image_id", "subject_idx"]
    feature_cols = [c for c in df.columns if c not in drop]
    X = df[feature_cols].fillna(0)
    y = df["class"]

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n  ✓ Test Accuracy : {acc:.4f}")
    print(f"  ✓ Test AUC-ROC  : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["TD", "ASD"]))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5-fold CV
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"\n  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Retrain on full data
    model.fit(X_scaled, y, verbose=False)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top Feature Importances:")
    for feat, imp in sorted_imp[:12]:
        bar = "█" * int(imp * 40)
        print(f"    {feat:25s}: {imp:.4f} {bar}")

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  ✓ Model saved: {MODEL_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: {SCALER_PATH}")

    metadata = {
        "model_type": "XGBClassifier",
        "dataset": "Saliency4ASD (300 images × 14 ASD + 14 TD subjects)",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "test_accuracy": round(acc, 4),
        "test_auc": round(auc, 4),
        "cv5_auc_mean": round(cv_scores.mean(), 4),
        "feature_importance": {k: round(v, 4) for k, v in sorted_imp[:15]},
        "note": "Gaze fixation pattern classifier for ASD vs TD. Scanpath-level features.",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {METADATA_PATH}")

    return model, scaler, feature_cols, metadata


if __name__ == "__main__":
    print("=" * 60)
    print("Eye-Tracking Gaze Pattern Model — Training")
    print("=" * 60)
    df = build_dataset()
    model, scaler, feature_cols, metadata = train_model(df)
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Test Accuracy: {metadata['test_accuracy']}")
    print(f"  Test AUC-ROC:  {metadata['test_auc']}")
    print(f"  5-Fold CV AUC: {metadata['cv5_auc_mean']}")
    print(f"{'='*60}")
