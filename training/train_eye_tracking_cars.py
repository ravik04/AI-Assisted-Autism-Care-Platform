"""
Eye-Tracking CARS Severity Enrichment Model
============================================
Uses MMASD Tobii eye-tracking CSV data (25 ASD participants) to predict
CARS severity scores from aggregate gaze features.

This enrichment layer adds clinical depth: given raw eye-tracking data,
predict the autism severity level (CARS score range: 17-45).

Output:
  - saved_models/eye_tracking_cars_model.pkl  (Ridge regression)
  - saved_models/eye_tracking_cars_scaler.pkl
  - saved_models/eye_tracking_cars_metadata.json
"""

import os, sys, json, time, warnings, gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# Paths
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET_DIR = r"D:\Autism\AutismData\Eye-Tracking Dataset\Eye-tracking Output"
META_PATH = r"D:\Autism\AutismData\Eye-Tracking Dataset\Metadata_Participants.csv"
SAVE_DIR = os.path.join(PROJECT, "saved_models")

LOG = os.path.join(PROJECT, "et_cars_training_log.txt")

def log(msg):
    line = f"  {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def extract_gaze_features(df):
    """Extract aggregate features from a single participant's Tobii CSV."""
    features = {}

    # Clean numeric columns
    numeric_cols = [
        'Pupil Diameter Right [mm]', 'Pupil Diameter Left [mm]',
        'Point of Regard Right X [px]', 'Point of Regard Right Y [px]',
        'Point of Regard Left X [px]', 'Point of Regard Left Y [px]',
        'Gaze Vector Right X', 'Gaze Vector Right Y', 'Gaze Vector Right Z',
        'Gaze Vector Left X', 'Gaze Vector Left Y', 'Gaze Vector Left Z',
        'RecordingTime [ms]',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── 1. Pupil diameter features ───────────────────────────────
    for side in ['Right', 'Left']:
        col = f'Pupil Diameter {side} [mm]'
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                features[f'pupil_{side.lower()}_mean'] = vals.mean()
                features[f'pupil_{side.lower()}_std'] = vals.std()
                features[f'pupil_{side.lower()}_median'] = vals.median()
                features[f'pupil_{side.lower()}_range'] = vals.max() - vals.min()
            else:
                for s in ['mean', 'std', 'median', 'range']:
                    features[f'pupil_{side.lower()}_{s}'] = 0.0

    # Average pupil
    r = df.get('Pupil Diameter Right [mm]', pd.Series(dtype=float))
    l = df.get('Pupil Diameter Left [mm]', pd.Series(dtype=float))
    r = pd.to_numeric(r, errors='coerce')
    l = pd.to_numeric(l, errors='coerce')
    avg_pupil = (r + l) / 2
    valid = avg_pupil.dropna()
    if len(valid) > 0:
        features['pupil_avg_mean'] = valid.mean()
        features['pupil_avg_std'] = valid.std()
    else:
        features['pupil_avg_mean'] = 0.0
        features['pupil_avg_std'] = 0.0

    # ── 2. Gaze position features ────────────────────────────────
    for side in ['Right', 'Left']:
        x_col = f'Point of Regard {side} X [px]'
        y_col = f'Point of Regard {side} Y [px]'
        if x_col in df.columns and y_col in df.columns:
            x = df[x_col].dropna()
            y = df[y_col].dropna()
            if len(x) > 10:
                features[f'gaze_{side.lower()}_x_mean'] = x.mean()
                features[f'gaze_{side.lower()}_x_std'] = x.std()
                features[f'gaze_{side.lower()}_y_mean'] = y.mean()
                features[f'gaze_{side.lower()}_y_std'] = y.std()
                # Gaze dispersion
                features[f'gaze_{side.lower()}_dispersion'] = np.sqrt(x.std()**2 + y.std()**2)
            else:
                for s in ['x_mean', 'x_std', 'y_mean', 'y_std', 'dispersion']:
                    features[f'gaze_{side.lower()}_{s}'] = 0.0

    # ── 3. Fixation/Saccade category analysis ────────────────────
    for side in ['Right', 'Left']:
        cat_col = f'Category {side}'
        if cat_col in df.columns:
            cats = df[cat_col].dropna()
            total = len(cats)
            if total > 0:
                fix_count = (cats == 'Fixation').sum()
                sac_count = (cats == 'Saccade').sum()
                features[f'fixation_ratio_{side.lower()}'] = fix_count / total
                features[f'saccade_ratio_{side.lower()}'] = sac_count / total
            else:
                features[f'fixation_ratio_{side.lower()}'] = 0.0
                features[f'saccade_ratio_{side.lower()}'] = 0.0

    # ── 4. Fixation duration features ────────────────────────────
    time_col = 'RecordingTime [ms]'
    cat_col_r = 'Category Right'
    if time_col in df.columns and cat_col_r in df.columns:
        fix_mask = df[cat_col_r] == 'Fixation'
        fix_times = df.loc[fix_mask, time_col].dropna()
        if len(fix_times) > 2:
            diffs = fix_times.diff().dropna()
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                features['fix_dur_mean'] = diffs.mean()
                features['fix_dur_std'] = diffs.std()
                features['fix_dur_median'] = diffs.median()
                features['fix_dur_max'] = diffs.max()
            else:
                for s in ['fix_dur_mean', 'fix_dur_std', 'fix_dur_median', 'fix_dur_max']:
                    features[s] = 0.0
        else:
            for s in ['fix_dur_mean', 'fix_dur_std', 'fix_dur_median', 'fix_dur_max']:
                features[s] = 0.0

    # ── 5. Saccade amplitude features ────────────────────────────
    if time_col in df.columns and cat_col_r in df.columns:
        sac_mask = df[cat_col_r] == 'Saccade'
        x_col = 'Point of Regard Right X [px]'
        y_col = 'Point of Regard Right Y [px]'
        if x_col in df.columns and y_col in df.columns:
            sac_x = df.loc[sac_mask, x_col].dropna().values
            sac_y = df.loc[sac_mask, y_col].dropna().values
            min_len = min(len(sac_x), len(sac_y))
            if min_len > 2:
                sac_x = sac_x[:min_len]
                sac_y = sac_y[:min_len]
                dx = np.diff(sac_x)
                dy = np.diff(sac_y)
                amps = np.sqrt(dx**2 + dy**2)
                features['sac_amp_mean'] = amps.mean()
                features['sac_amp_std'] = amps.std()
                features['sac_amp_max'] = amps.max()
            else:
                features['sac_amp_mean'] = 0.0
                features['sac_amp_std'] = 0.0
                features['sac_amp_max'] = 0.0
        else:
            features['sac_amp_mean'] = 0.0
            features['sac_amp_std'] = 0.0
            features['sac_amp_max'] = 0.0

    # ── 6. Gaze velocity ─────────────────────────────────────────
    x_col = 'Point of Regard Right X [px]'
    y_col = 'Point of Regard Right Y [px]'
    if x_col in df.columns and y_col in df.columns and time_col in df.columns:
        valid_idx = df[[x_col, y_col, time_col]].dropna().index
        if len(valid_idx) > 10:
            x = df.loc[valid_idx, x_col].values
            y = df.loc[valid_idx, y_col].values
            t = df.loc[valid_idx, time_col].values
            dt = np.diff(t)
            dt[dt == 0] = 1e-6
            dx = np.diff(x)
            dy = np.diff(y)
            vel = np.sqrt(dx**2 + dy**2) / dt
            vel = vel[np.isfinite(vel)]
            if len(vel) > 0:
                features['gaze_vel_mean'] = vel.mean()
                features['gaze_vel_std'] = vel.std()
                features['gaze_vel_median'] = np.median(vel)
                features['gaze_vel_p90'] = np.percentile(vel, 90)
            else:
                for s in ['gaze_vel_mean', 'gaze_vel_std', 'gaze_vel_median', 'gaze_vel_p90']:
                    features[s] = 0.0
        else:
            for s in ['gaze_vel_mean', 'gaze_vel_std', 'gaze_vel_median', 'gaze_vel_p90']:
                features[s] = 0.0

    # ── 7. AOI (Area of Interest) features ───────────────────────
    for side in ['Right', 'Left']:
        aoi_col = f'AOI Name {side}'
        if aoi_col in df.columns:
            aois = df[aoi_col].dropna()
            if len(aois) > 0:
                features[f'aoi_{side.lower()}_unique'] = aois.nunique()
                # Proportion in most common AOI
                top_aoi = aois.value_counts().iloc[0] / len(aois)
                features[f'aoi_{side.lower()}_top_ratio'] = top_aoi
            else:
                features[f'aoi_{side.lower()}_unique'] = 0
                features[f'aoi_{side.lower()}_top_ratio'] = 0.0

    # ── 8. Tracking ratio ────────────────────────────────────────
    if 'Tracking Ratio [%]' in df.columns:
        tr = pd.to_numeric(df['Tracking Ratio [%]'], errors='coerce').dropna()
        if len(tr) > 0:
            features['tracking_ratio_mean'] = tr.mean()
        else:
            features['tracking_ratio_mean'] = 0.0

    # ── 9. Recording duration ────────────────────────────────────
    if time_col in df.columns:
        times = df[time_col].dropna()
        if len(times) > 1:
            features['total_recording_ms'] = times.max() - times.min()
            features['n_samples'] = len(times)
        else:
            features['total_recording_ms'] = 0.0
            features['n_samples'] = 0

    # ── 10. Gaze direction features (3D vectors) ─────────────────
    for side in ['Right', 'Left']:
        for axis in ['X', 'Y', 'Z']:
            col = f'Gaze Vector {side} {axis}'
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    features[f'gvec_{side.lower()}_{axis.lower()}_mean'] = vals.mean()
                    features[f'gvec_{side.lower()}_{axis.lower()}_std'] = vals.std()
                else:
                    features[f'gvec_{side.lower()}_{axis.lower()}_mean'] = 0.0
                    features[f'gvec_{side.lower()}_{axis.lower()}_std'] = 0.0

    return features


def main():
    t0 = time.time()
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Eye-Tracking CARS Severity Enrichment Model\n")
        f.write("=" * 60 + "\n\n")

    # ── 1. Load metadata ─────────────────────────────────────────
    log("[1/5] Loading metadata...")
    meta = pd.read_csv(META_PATH)
    # Keep only ASD with CARS scores
    asd_meta = meta[(meta['Class'] == 'ASD') & (meta['CARS Score'].notna())].copy()
    log(f"  ASD participants with CARS: {len(asd_meta)}")
    log(f"  CARS range: {asd_meta['CARS Score'].min()} - {asd_meta['CARS Score'].max()}")

    # ── 2. Extract features from each CSV ────────────────────────
    log("\n[2/5] Extracting gaze features from CSVs...")
    all_features = []
    all_cars = []
    all_pids = []

    for _, row in asd_meta.iterrows():
        pid = int(row['ParticipantID'])
        cars = row['CARS Score']
        csv_path = os.path.join(ET_DIR, f"{pid}.csv")

        if not os.path.exists(csv_path):
            log(f"  Skipping PID {pid} — CSV not found")
            continue

        try:
            # Read only first 50000 rows to keep things fast
            df = pd.read_csv(csv_path, nrows=50000, low_memory=False)
            feats = extract_gaze_features(df)
            all_features.append(feats)
            all_cars.append(cars)
            all_pids.append(pid)
            log(f"  PID {pid}: CARS={cars}, {len(feats)} features extracted")
            del df
            gc.collect()
        except Exception as e:
            log(f"  Error PID {pid}: {e}")
            continue

    log(f"\n  Total samples: {len(all_features)}")

    if len(all_features) < 5:
        log("ERROR: Not enough samples!")
        return

    # ── 3. Build feature matrix ──────────────────────────────────
    log("\n[3/5] Building feature matrix...")
    X_df = pd.DataFrame(all_features).fillna(0)
    y = np.array(all_cars)
    pids = np.array(all_pids)

    # Remove zero-variance features
    variances = X_df.var()
    keep_cols = variances[variances > 1e-10].index.tolist()
    X_df = X_df[keep_cols]
    log(f"  Features after variance filter: {len(keep_cols)}")
    log(f"  Samples: {len(y)}, CARS mean={y.mean():.1f}, std={y.std():.1f}")

    X = X_df.values.astype(np.float32)

    # ── 4. Train with Leave-One-Out CV ───────────────────────────
    log("\n[4/5] Training Ridge regression with LOO-CV...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try multiple alphas
    best_alpha = 1.0
    best_mae = 999
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        loo = LeaveOneOut()
        preds = cross_val_predict(model, X_scaled, y, cv=loo)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        log(f"  alpha={alpha:>7.1f} → MAE={mae:.2f}, R²={r2:.3f}")
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    log(f"\n  Best alpha: {best_alpha}")

    # Final model
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_scaled, y)

    # LOO predictions for final metrics
    loo_preds = cross_val_predict(final_model, X_scaled, y, cv=LeaveOneOut())
    final_mae = mean_absolute_error(y, loo_preds)
    final_r2 = r2_score(y, loo_preds)

    log(f"  Final: MAE={final_mae:.2f}, R²={final_r2:.3f}")
    log(f"  Prediction range: {loo_preds.min():.1f} - {loo_preds.max():.1f}")

    # CARS severity classification (< 30 mild, 30-36 moderate, > 36 severe)
    y_cat = np.where(y < 30, 'mild', np.where(y <= 36, 'moderate', 'severe'))
    p_cat = np.where(loo_preds < 30, 'mild', np.where(loo_preds <= 36, 'moderate', 'severe'))
    sev_acc = (y_cat == p_cat).mean()
    log(f"  Severity category accuracy: {sev_acc:.1%}")

    # ── 5. Save ──────────────────────────────────────────────────
    log("\n[5/5] Saving model...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    joblib.dump(final_model, os.path.join(SAVE_DIR, "eye_tracking_cars_model.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "eye_tracking_cars_scaler.pkl"))

    metadata = {
        "model_type": "Ridge Regression",
        "task": "CARS severity prediction from Tobii eye-tracking",
        "dataset": "MMASD Eye-Tracking Output (25 ASD participants)",
        "n_samples": len(y),
        "n_features": X.shape[1],
        "feature_names": keep_cols,
        "alpha": best_alpha,
        "loo_mae": round(float(final_mae), 3),
        "loo_r2": round(float(final_r2), 3),
        "severity_accuracy": round(float(sev_acc), 3),
        "cars_range": {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean())},
        "severity_thresholds": {"mild": "<30", "moderate": "30-36", "severe": ">36"},
        "participant_predictions": {
            str(int(pid)): {"actual": float(act), "predicted": round(float(pred), 1)}
            for pid, act, pred in zip(pids, y, loo_preds)
        }
    }
    with open(os.path.join(SAVE_DIR, "eye_tracking_cars_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"DONE! {len(y)} participants, {X.shape[1]} features")
    log(f"LOO-CV: MAE={final_mae:.2f}, R²={final_r2:.3f}")
    log(f"Severity accuracy: {sev_acc:.1%}")
    log(f"Time: {elapsed:.0f}s")
    log(f"Saved: eye_tracking_cars_model.pkl, eye_tracking_cars_scaler.pkl")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
