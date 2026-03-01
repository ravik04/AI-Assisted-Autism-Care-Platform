"""Peek at MMASD skeleton data structure to understand formats."""
import os
import json
import numpy as np
import sys

OUT_FILE = r'C:\Users\ravik\AutismCare\autism_ai_prototype_v2\mmasd_peek_output.txt'
sys.stdout = open(OUT_FILE, 'w', encoding='utf-8')

MMASD_ROOT = r'D:\Autism\AutismData\MMASD\Dataset_FINAL'

# 1. Peek at OpenPose JSON
openpose_dir = os.path.join(MMASD_ROOT, '2D skeleton', '2D_openpose', 'output')
print("=== OPENPOSE 2D DATA ===")
activities = os.listdir(openpose_dir)
print(f"Activities ({len(activities)}): {activities}")

# Pick first activity, first clip
first_act = activities[0]
clips = os.listdir(os.path.join(openpose_dir, first_act))
print(f"\nActivity '{first_act}' has {len(clips)} clips")
print(f"Sample clips: {clips[:5]}")

# Look at naming convention
# e.g. as_20583_D16_000_i
# as = arm_swing, 20583 = participant ID, D16 = session?, 000 = clip, i/y = ?
print(f"\nNaming pattern analysis:")
for c in clips[:5]:
    parts = c.split('_')
    print(f"  {c} -> parts: {parts}")

# Read first json keypoints
first_clip = clips[0]
frames_dir = os.path.join(openpose_dir, first_act, first_clip)
frame_files = sorted(os.listdir(frames_dir))
print(f"\nClip '{first_clip}' has {len(frame_files)} frames")

with open(os.path.join(frames_dir, frame_files[0]), 'r') as f:
    data = json.load(f)
print(f"\nOpenPose JSON keys: {data.keys()}")
print(f"Version: {data['version']}")
print(f"People detected: {len(data['people'])}")
if len(data['people']) > 0:
    person = data['people'][0]
    print(f"\nPerson keys: {list(person.keys())}")
    for k, v in person.items():
        if isinstance(v, list):
            print(f"  {k}: list len={len(v)}, first 10 vals: {v[:10]}")
        else:
            print(f"  {k}: {v}")

# 2. Peek at ROMP 2D NPZ data
romp_2d_dir = os.path.join(MMASD_ROOT, '2D skeleton', 'ROMP_2D_Coordinates')
print("\n\n=== ROMP 2D COORDINATES ===")
activities_romp = os.listdir(romp_2d_dir)
print(f"Activities ({len(activities_romp)}): {activities_romp}")

act1 = activities_romp[0]
clips_romp = os.listdir(os.path.join(romp_2d_dir, act1))
print(f"\nActivity '{act1}' has {len(clips_romp)} clips")
print(f"Sample clips: {clips_romp[:5]}")

# Read a sample NPZ
first_clip_romp = clips_romp[0]
clip_dir_romp = os.path.join(romp_2d_dir, act1, first_clip_romp)
frame_files_romp = sorted(os.listdir(clip_dir_romp))
print(f"\nClip '{first_clip_romp}' has {len(frame_files_romp)} frames")

npz_path = os.path.join(clip_dir_romp, frame_files_romp[0])
npz_data = np.load(npz_path, allow_pickle=True)
print(f"\nNPZ keys: {list(npz_data.keys())}")
for k in npz_data.keys():
    arr = npz_data[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size < 200:
        print(f"    values: {arr}")

# 3. Count total clips per activity & participant mapping
print("\n\n=== ACTIVITY-LEVEL STATISTICS (ROMP 2D) ===")
all_participant_ids = set()
for act in activities_romp:
    act_dir = os.path.join(romp_2d_dir, act)
    clips = os.listdir(act_dir)
    # Extract participant IDs from clip names
    pids = set()
    for c in clips:
        parts = c.split('_')
        if len(parts) >= 2:
            # Format: as_20583_D1_000_y
            pid = parts[1]  # participant ID
            pids.add(pid)
            all_participant_ids.add(pid)
    frame_counts = []
    for c in clips[:5]:
        cdir = os.path.join(act_dir, c)
        if os.path.isdir(cdir):
            frame_counts.append(len(os.listdir(cdir)))
    print(f"  {act}: {len(clips)} clips, {len(pids)} participants, sample frame counts: {frame_counts}")

print(f"\nTotal unique participants found in ROMP 2D: {len(all_participant_ids)}")
print(f"Participant IDs: {sorted(all_participant_ids)}")

# 4. Check ROMP 3D for comparison
romp_3d_dir = os.path.join(MMASD_ROOT, '3D skeleton', 'ROMP_3D_Coordinates')
if os.path.exists(romp_3d_dir):
    print("\n\n=== ROMP 3D COORDINATES ===")
    act1_3d = os.listdir(romp_3d_dir)[0]
    clips_3d = os.listdir(os.path.join(romp_3d_dir, act1_3d))
    clip_dir_3d = os.path.join(romp_3d_dir, act1_3d, clips_3d[0])
    frame_files_3d = sorted(os.listdir(clip_dir_3d))
    npz_path_3d = os.path.join(clip_dir_3d, frame_files_3d[0])
    npz_data_3d = np.load(npz_path_3d, allow_pickle=True)
    print(f"3D NPZ keys: {list(npz_data_3d.keys())}")
    for k in npz_data_3d.keys():
        arr = npz_data_3d[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.size < 100:
            print(f"    values: {arr}")

# 5. Also peek at Eye-Tracking CSV
print("\n\n=== EYE-TRACKING DATASET ===")
et_dir = r'D:\Autism\AutismData\Eye-Tracking Dataset'
if os.path.exists(et_dir):
    files = os.listdir(et_dir)
    print(f"Files ({len(files)}): {files[:15]}")
    
    # Read metadata
    import pandas as pd
    meta_path = os.path.join(et_dir, 'Metadata_Participants.csv')
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        print(f"\nMetadata shape: {meta.shape}")
        print(f"Columns: {meta.columns.tolist()}")
        print(f"\nGroup distribution:")
        print(meta['Group'].value_counts().to_string())
        print(f"\nSample rows:")
        print(meta.head(5).to_string())
    
    # Read a sample ASD CSV
    csv_files = [f for f in files if f.endswith('.csv') and f != 'Metadata_Participants.csv']
    if csv_files:
        sample_csv = os.path.join(et_dir, csv_files[0])
        df_sample = pd.read_csv(sample_csv, sep='\t', nrows=10, on_bad_lines='skip')
        print(f"\nSample CSV: {csv_files[0]}")
        print(f"Columns ({len(df_sample.columns)}): {df_sample.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(df_sample.head(3).to_string())

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Done. Output in {OUT_FILE}")
