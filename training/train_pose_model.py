"""
Train Pose/Skeleton Model on MMASD Dataset (Optimized)
========================================================
Uses ROMP 2D skeleton coordinates (24 joints x 2 per frame) from video clips
of children performing 11 therapy activities.

Key optimizations:
- Subsample frames during loading (every Nth) instead of loading all
- Batch-process with progress tracking
- Lightweight feature extraction focused on most discriminative features
"""

import os
import sys
import gc
import json
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
MMASD_ROOT = Path(r'D:\Autism\AutismData\MMASD\Dataset_FINAL')
ROMP_2D_DIR = MMASD_ROOT / '2D skeleton' / 'ROMP_2D_Coordinates'
ADOS_FILE = MMASD_ROOT / 'ADOS_rating.xlsx'
SAVE_DIR = Path(r'C:\Users\ravik\AutismCare\autism_ai_prototype_v2\saved_models')
SAVE_DIR.mkdir(exist_ok=True)
LOG_FILE = Path(r'C:\Users\ravik\AutismCare\autism_ai_prototype_v2\pose_training_log.txt')

# Performance settings — aggressive to avoid D: drive I/O death
TARGET_FRAMES = 30      # Sample 30 frames per clip (fast + enough signal)
MAX_LOAD_FRAMES = 30    # Only load this many frames (subsample during read)
MIN_FRAMES = 15         # Minimum frames to consider clip valid
MAX_CLIPS_PER_PID = 15  # Limit clips per participant to reduce total I/O

# Joint definitions (SMPL 24-joint)
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1',
    'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3',
    'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

SYMMETRIC_PAIRS = [
    (1, 2), (4, 5), (7, 8), (10, 11),
    (13, 14), (16, 17), (18, 19), (20, 21), (22, 23),
]

ANGLE_TRIPLETS = [
    (16, 18, 20), (17, 19, 21),  # shoulder-elbow-wrist
    (1, 4, 7), (2, 5, 8),        # hip-knee-ankle
    (13, 16, 18), (14, 17, 19),  # collar-shoulder-elbow
    (0, 3, 6), (6, 9, 12),       # spine chain
]

# ============================================================================
# Logging
# ============================================================================
log_fh = open(LOG_FILE, 'w', encoding='utf-8')

def log(msg):
    """Print and log simultaneously."""
    print(msg, flush=True)
    log_fh.write(msg + '\n')
    log_fh.flush()

# ============================================================================
# Step 1: Load ADOS Labels
# ============================================================================
log("=" * 70)
log("MMASD Pose/Skeleton Model Training (Optimized)")
log("=" * 70)

log("\n[1/7] Loading ADOS clinical ratings...")
ados_df = pd.read_excel(ADOS_FILE)
ados_df = ados_df.dropna(subset=['Gender'])
ados_df = ados_df[ados_df['Gender'].isin(['M', 'F'])]

severity_col = [c for c in ados_df.columns if 'Severity' in c][0]
comparison_col = [c for c in ados_df.columns if 'Comparison' in c][0]
social_col = [c for c in ados_df.columns if 'Social Affect' in c][0]
rrb_col = [c for c in ados_df.columns if 'RRB' in c][0]

ados_participants = {}
for _, row in ados_df.iterrows():
    pid = str(row['ID#']).replace('_R', '')
    ados_participants[pid] = {
        'severity': int(row[severity_col]),
        'comparison_score': int(row[comparison_col]),
        'social_affect': float(row[social_col]),
        'rrb': float(row[rrb_col]),
    }

log(f"  ADOS participants: {len(ados_participants)}")

# ============================================================================
# Step 2: Discover clips
# ============================================================================
log("\n[2/7] Discovering clips...")
activities = sorted([d for d in os.listdir(ROMP_2D_DIR) if (ROMP_2D_DIR / d).is_dir()])
activity_encoding = {act: i for i, act in enumerate(activities)}

clip_registry = []
pid_clip_counts = {}  # Track clips per participant to enforce limit

for act in activities:
    act_dir = ROMP_2D_DIR / act
    clips = sorted([d for d in os.listdir(act_dir) if (act_dir / d).is_dir()])
    for clip_name in clips:
        parts = clip_name.split('_')
        if len(parts) < 2:
            continue
        pid = parts[1]
        if pid in ados_participants:
            label = 1  # ASD
        elif pid.startswith('41'):
            label = 0  # TD
        else:
            continue
        
        # Enforce per-participant clip limit
        pid_clip_counts.setdefault(pid, 0)
        if pid_clip_counts[pid] >= MAX_CLIPS_PER_PID:
            continue
        
        clip_path = act_dir / clip_name
        npz_files = [f for f in os.listdir(clip_path) if f.endswith('.npz')]
        if len(npz_files) >= MIN_FRAMES:
            clip_registry.append({
                'path': str(clip_path),
                'pid': pid,
                'activity': act,
                'activity_id': activity_encoding[act],
                'label': label,
                'n_frames': len(npz_files),
                'clip_name': clip_name,
            })
            pid_clip_counts[pid] += 1

log(f"  Total valid clips: {len(clip_registry)}")
log(f"  ASD={sum(c['label']==1 for c in clip_registry)}, TD={sum(c['label']==0 for c in clip_registry)}")
log(f"  Participants: ASD={len(set(c['pid'] for c in clip_registry if c['label']==1))}, TD={len(set(c['pid'] for c in clip_registry if c['label']==0))}")

# ============================================================================
# Step 3: Fast skeleton loader
# ============================================================================
log("\n[3/7] Defining optimized feature pipeline...")

def load_clip_fast(clip_path, max_frames=MAX_LOAD_FRAMES):
    """Load skeleton frames, subsampling to at most max_frames."""
    frame_files = sorted([f for f in os.listdir(clip_path) if f.endswith('.npz')])
    n = len(frame_files)
    if n < MIN_FRAMES:
        return None
    
    # Subsample: pick evenly spaced frame indices
    if n > max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    frames = []
    for ff in frame_files:
        try:
            npz = np.load(os.path.join(clip_path, ff), allow_pickle=True)
            coords = npz['coordinates']  # (N_people, 24, 2)
            if coords.shape[0] == 0:
                continue
            # Pick first person (or closest to center)
            if coords.shape[0] == 1:
                person = coords[0]
            else:
                centers = coords.mean(axis=1)
                dists = np.linalg.norm(centers - np.array([320, 240]), axis=1)
                person = coords[np.argmin(dists)]
            frames.append(person)
        except:
            continue
    
    if len(frames) < MIN_FRAMES:
        return None
    
    seq = np.array(frames)  # (T, 24, 2)
    
    # Normalize: center on pelvis
    pelvis = seq[:, 0:1, :]
    seq = seq - pelvis
    
    # Scale by torso length
    torso = np.linalg.norm(seq[:, 12, :] - seq[:, 0, :], axis=1)
    torso = np.maximum(torso, 1e-8)
    seq = seq / torso[:, None, None]
    
    # Resample to exact target frames
    if len(seq) != TARGET_FRAMES:
        idx = np.linspace(0, len(seq) - 1, TARGET_FRAMES, dtype=int)
        seq = seq[idx]
    
    return seq


def compute_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_a, -1, 1)) * 180 / np.pi


def extract_features(seq, activity_id):
    """Focused feature extraction — ~350 features."""
    T, J, D = seq.shape
    feats = {}
    
    # 1. Position stats (mean, std, range per joint) — 6*24 = 144 feats
    for j in range(J):
        pos = seq[:, j, :]
        feats[f'j{j}_mx'] = np.mean(pos[:, 0])
        feats[f'j{j}_my'] = np.mean(pos[:, 1])
        feats[f'j{j}_sx'] = np.std(pos[:, 0])
        feats[f'j{j}_sy'] = np.std(pos[:, 1])
        feats[f'j{j}_rx'] = np.ptp(pos[:, 0])
        feats[f'j{j}_ry'] = np.ptp(pos[:, 1])
    
    # 2. Velocity stats — aggregate + per-key-joint
    vel = np.diff(seq, axis=0)
    speed = np.linalg.norm(vel, axis=2)  # (T-1, 24)
    
    feats['speed_mean'] = np.mean(speed)
    feats['speed_std'] = np.std(speed)
    feats['speed_max'] = np.max(speed)
    
    key_joints = [15, 20, 21, 22, 23, 7, 8]  # head, wrists, hands, ankles
    for j in key_joints:
        feats[f'sp{j}_m'] = np.mean(speed[:, j])
        feats[f'sp{j}_x'] = np.max(speed[:, j])
        feats[f'sp{j}_s'] = np.std(speed[:, j])
    
    # 3. Acceleration
    accel = np.diff(vel, axis=0)
    accel_mag = np.linalg.norm(accel, axis=2)
    feats['accel_mean'] = np.mean(accel_mag)
    feats['accel_std'] = np.std(accel_mag)
    
    # 4. Jerk (smoothness)
    if len(accel) > 1:
        jerk = np.diff(accel, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=2)
        feats['jerk_mean'] = np.mean(jerk_mag)
        feats['jerk_ratio'] = np.mean(jerk_mag) / (feats['speed_mean'] + 1e-8)
    else:
        feats['jerk_mean'] = 0
        feats['jerk_ratio'] = 0
    
    # 5. Symmetry — 9 pairs × 3 feats = 27
    for li, ri in SYMMETRIC_PAIRS:
        lp, rp = seq[:, li, :], seq[:, ri, :]
        rm = rp.copy(); rm[:, 0] = -rm[:, 0]
        sym_diff = np.linalg.norm(lp - rm, axis=1)
        feats[f'sym_{li}_{ri}_m'] = np.mean(sym_diff)
        feats[f'sym_{li}_{ri}_s'] = np.std(sym_diff)
        feats[f'spdsy_{li}_{ri}'] = np.mean(np.abs(speed[:, li] - speed[:, ri]))
    
    feats['sym_overall'] = np.mean([feats[f'sym_{li}_{ri}_m'] for li, ri in SYMMETRIC_PAIRS])
    
    # 6. Joint angles — 8 triplets × 3 feats = 24
    for ti, (j1, j2, j3) in enumerate(ANGLE_TRIPLETS):
        angles = np.array([compute_angle(seq[t, j1], seq[t, j2], seq[t, j3]) for t in range(T)])
        feats[f'ang{ti}_m'] = np.mean(angles)
        feats[f'ang{ti}_s'] = np.std(angles)
        feats[f'ang{ti}_r'] = np.ptp(angles)
    
    # 7. Repetitive motion (autocorrelation of hands)
    for j in [20, 21, 22, 23]:
        sp = speed[:, j]
        sp_centered = sp - np.mean(sp)
        sp_std = np.std(sp)
        if sp_std > 1e-8 and len(sp) > 10:
            ac = np.correlate(sp_centered / sp_std, sp_centered / sp_std, mode='full')
            ac = ac[len(sp)-1:len(sp)+min(20, len(sp)//2)] / len(sp)
            feats[f'ac_peak_{j}'] = np.max(ac[1:]) if len(ac) > 1 else 0
        else:
            feats[f'ac_peak_{j}'] = 0
    
    # 8. Center of mass
    com = np.mean(seq, axis=1)  # (T, 2)
    com_sp = np.linalg.norm(np.diff(com, axis=0), axis=1)
    feats['com_disp'] = np.linalg.norm(com[-1] - com[0])
    feats['com_dist'] = np.sum(com_sp)
    feats['com_speed'] = np.mean(com_sp)
    feats['com_spread'] = np.mean(np.std(com, axis=0))
    
    # 9. Body bounding box
    widths = np.ptp(seq[:, :, 0], axis=1)
    heights = np.ptp(seq[:, :, 1], axis=1)
    feats['body_w_m'] = np.mean(widths)
    feats['body_w_s'] = np.std(widths)
    feats['body_h_m'] = np.mean(heights)
    feats['body_h_s'] = np.std(heights)
    
    # 10. Head movement
    head = seq[:, 15, :]
    head_sp = np.linalg.norm(np.diff(head, axis=0), axis=1)
    feats['head_sp_m'] = np.mean(head_sp)
    feats['head_sp_x'] = np.max(head_sp)
    feats['head_rx'] = np.ptp(head[:, 0])
    feats['head_ry'] = np.ptp(head[:, 1])
    
    # 11. Activity type
    feats['activity'] = activity_id
    
    # 12. Limb lengths (stability)
    for j1, j2 in [(16,18), (17,19), (18,20), (19,21), (1,4), (2,5), (4,7), (5,8)]:
        ll = np.linalg.norm(seq[:, j1] - seq[:, j2], axis=1)
        feats[f'limb_{j1}_{j2}_m'] = np.mean(ll)
        feats[f'limb_{j1}_{j2}_s'] = np.std(ll)
    
    return feats


# ============================================================================
# Step 4: Extract features
# ============================================================================
log("\n[4/7] Extracting features from all clips...")
t0 = time.time()

all_features = []
all_labels = []
all_pids = []
failed = 0

for idx, clip in enumerate(clip_registry):
    if idx % 50 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / max(elapsed, 0.1)
        eta = (len(clip_registry) - idx) / max(rate, 0.01)
        log(f"  [{idx+1}/{len(clip_registry)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
        gc.collect()  # Free memory periodically
    
    try:
        seq = load_clip_fast(clip['path'])
        if seq is None:
            failed += 1
            continue
        
        feats = extract_features(seq, clip['activity_id'])
        all_features.append(feats)
        all_labels.append(clip['label'])
        all_pids.append(clip['pid'])
        del seq  # Free immediately
    except Exception as e:
        failed += 1
        if failed <= 5:
            log(f"    Error on {clip['clip_name']}: {e}")

elapsed = time.time() - t0
log(f"  Done! {len(all_features)} clips in {elapsed:.1f}s ({failed} failed)")

# Build matrices
X_df = pd.DataFrame(all_features).replace([np.inf, -np.inf], np.nan).fillna(0)
y = np.array(all_labels)
pids = np.array(all_pids)
feature_names = X_df.columns.tolist()
X = X_df.values

log(f"  Feature matrix: {X.shape}")
log(f"  Labels: ASD={sum(y==1)}, TD={sum(y==0)}")

# ============================================================================
# Step 5: Train with GroupKFold (participant-aware)
# ============================================================================
log("\n[5/7] Training XGBoost with 5-fold GroupKFold CV...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gkf = GroupKFold(n_splits=5)
fold_metrics = []
best_model = None
best_auc = 0
oof_preds = np.zeros(len(y))
oof_probs = np.zeros(len(y))

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_scaled, y, groups=pids)):
    Xtr, Xva = X_scaled[tr_idx], X_scaled[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    
    n_pos, n_neg = sum(ytr == 1), sum(ytr == 0)
    spw = n_neg / max(n_pos, 1) if n_pos < n_neg else 1.0
    
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=42,
        eval_metric='logloss', use_label_encoder=False,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    
    vp = model.predict_proba(Xva)[:, 1]
    vl = model.predict(Xva)
    
    acc = accuracy_score(yva, vl)
    f1 = f1_score(yva, vl, zero_division=0)
    try:
        auc = roc_auc_score(yva, vp)
    except:
        auc = 0.5
    
    fold_metrics.append({'accuracy': acc, 'f1': f1, 'auc': auc})
    oof_preds[va_idx] = vl
    oof_probs[va_idx] = vp
    
    if auc > best_auc:
        best_auc = auc
        best_model = model
    
    log(f"  Fold {fold+1}: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f} (val: {len(yva)} - ASD:{sum(yva==1)} TD:{sum(yva==0)})")

# Overall
oof_acc = accuracy_score(y, oof_preds)
oof_f1 = f1_score(y, oof_preds, zero_division=0)
try:
    oof_auc = roc_auc_score(y, oof_probs)
except:
    oof_auc = 0.5

log(f"\n  OOF Results: Acc={oof_acc:.4f} F1={oof_f1:.4f} AUC={oof_auc:.4f}")
log(f"\n  Mean CV: Acc={np.mean([m['accuracy'] for m in fold_metrics]):.4f}+/-{np.std([m['accuracy'] for m in fold_metrics]):.4f}")
log(f"           F1 ={np.mean([m['f1'] for m in fold_metrics]):.4f}+/-{np.std([m['f1'] for m in fold_metrics]):.4f}")
log(f"           AUC={np.mean([m['auc'] for m in fold_metrics]):.4f}+/-{np.std([m['auc'] for m in fold_metrics]):.4f}")

log(f"\n  Confusion Matrix:\n{confusion_matrix(y, oof_preds)}")
log(f"\n{classification_report(y, oof_preds, target_names=['TD', 'ASD'])}")

# ============================================================================
# Step 6: Final model
# ============================================================================
log("\n[6/7] Training final model on all data...")
n_pos, n_neg = sum(y == 1), sum(y == 0)
spw = n_neg / max(n_pos, 1) if n_pos < n_neg else 1.0

final_model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, random_state=42,
    eval_metric='logloss', use_label_encoder=False,
)
final_model.fit(X_scaled, y, verbose=False)

importances = final_model.feature_importances_
top_k = 20
top_idx = np.argsort(importances)[::-1][:top_k]
log(f"\n  Top {top_k} features:")
for rank, i in enumerate(top_idx):
    log(f"    {rank+1}. {feature_names[i]}: {importances[i]:.4f}")

# ============================================================================
# Step 7: Save
# ============================================================================
log("\n[7/7] Saving model and metadata...")

joblib.dump(final_model, SAVE_DIR / 'pose_skeleton_xgb.pkl')
joblib.dump(scaler, SAVE_DIR / 'pose_skeleton_scaler.pkl')

metadata = {
    'model_type': 'XGBClassifier',
    'task': 'ASD vs TD classification from skeleton/pose data',
    'dataset': 'MMASD',
    'data_source': 'ROMP 2D skeleton (24 SMPL joints)',
    'n_samples': int(len(y)),
    'n_features': int(len(feature_names)),
    'n_participants': {
        'ASD': int(len(set(p for p, l in zip(pids, y) if l == 1))),
        'TD': int(len(set(p for p, l in zip(pids, y) if l == 0)))
    },
    'activities': activities,
    'label_distribution': {'ASD': int(sum(y==1)), 'TD': int(sum(y==0))},
    'cv_results': {
        'accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'f1': float(np.mean([m['f1'] for m in fold_metrics])),
        'auc': float(np.mean([m['auc'] for m in fold_metrics])),
    },
    'oof_accuracy': float(oof_acc),
    'oof_auc': float(oof_auc),
    'feature_names': feature_names,
    'top_features': [
        {'name': feature_names[i], 'importance': float(importances[i])}
        for i in top_idx
    ],
    'joint_names': JOINT_NAMES,
}

with open(SAVE_DIR / 'pose_skeleton_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

log(f"\n{'='*70}")
log(f"DONE! {len(y)} clips, {len(np.unique(pids))} participants")
log(f"OOF: Acc={oof_acc:.1%} | AUC={oof_auc:.4f}")
log(f"Saved: pose_skeleton_xgb.pkl, pose_skeleton_scaler.pkl")
log(f"{'='*70}")

log_fh.close()
