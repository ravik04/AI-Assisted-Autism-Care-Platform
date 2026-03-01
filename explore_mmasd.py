"""Explore MMASD dataset structure for training pose model."""
import os
import json
import sys
import numpy as np

MMASD_ROOT = r'D:\Autism\AutismData\MMASD\Dataset_FINAL'
OUT_FILE = r'C:\Users\ravik\AutismCare\autism_ai_prototype_v2\mmasd_explore_output.txt'

# Redirect stdout to file
sys.stdout = open(OUT_FILE, 'w', encoding='utf-8')

print("=" * 60)
print("MMASD Dataset Explorer")
print("=" * 60)

# List top-level directories
print("\n--- Top-level contents ---")
for item in os.listdir(MMASD_ROOT):
    path = os.path.join(MMASD_ROOT, item)
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"  [DIR] {item} ({count} items)")
    else:
        size = os.path.getsize(path)
        print(f"  [FILE] {item} ({size:,} bytes)")

# Explore 2D skeleton
skel_2d = os.path.join(MMASD_ROOT, '2D skeleton')
print(f"\n--- 2D skeleton contents ---")
for item in os.listdir(skel_2d):
    path = os.path.join(skel_2d, item)
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"  [DIR] {item} ({count} items)")
    else:
        size = os.path.getsize(path)
        print(f"  [FILE] {item} ({size/1e6:.1f} MB)")

# Check if any ZIP files need extraction
import zipfile
for item in os.listdir(skel_2d):
    if item.endswith('.zip'):
        zpath = os.path.join(skel_2d, item)
        with zipfile.ZipFile(zpath, 'r') as zf:
            names = zf.namelist()
            print(f"\n--- ZIP: {item} ({len(names)} files) ---")
            for n in names[:15]:
                print(f"  {n}")
            if len(names) > 15:
                print(f"  ... and {len(names)-15} more")

# Check if there are already extracted folders
for item in os.listdir(skel_2d):
    path = os.path.join(skel_2d, item)
    if os.path.isdir(path):
        subitems = os.listdir(path)
        print(f"\n--- DIR: {item} ({len(subitems)} items) ---")
        for s in subitems[:10]:
            spath = os.path.join(path, s)
            if os.path.isdir(spath):
                count = len(os.listdir(spath))
                print(f"  [DIR] {s} ({count} items)")
            else:
                size = os.path.getsize(spath)
                print(f"  [FILE] {s} ({size:,} bytes)")
        if len(subitems) > 10:
            print(f"  ... and {len(subitems)-10} more")

# Try to peek at a sample data file
print("\n--- Looking for sample skeleton data ---")
for root, dirs, files in os.walk(skel_2d):
    for f in files:
        if f.endswith(('.json', '.csv', '.txt', '.npy')):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"\nFound: {fpath} ({size:,} bytes)")
            if f.endswith('.json'):
                with open(fpath, 'r') as fp:
                    data = json.load(fp)
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:10]}")
                    for k, v in list(data.items())[:2]:
                        if isinstance(v, list):
                            print(f"  {k}: list len={len(v)}")
                            if len(v) > 0 and isinstance(v[0], list):
                                print(f"    v[0] len={len(v[0])}, v[0][0]={v[0][0]}")
                        else:
                            print(f"  {k}: {type(v).__name__}")
                elif isinstance(data, list):
                    print(f"  List length: {len(data)}")
                    if len(data) > 0:
                        print(f"  First element type: {type(data[0]).__name__}")
            elif f.endswith('.csv'):
                with open(fpath, 'r') as fp:
                    lines = fp.readlines()[:5]
                for l in lines:
                    print(f"  {l.strip()[:200]}")
            elif f.endswith('.npy'):
                arr = np.load(fpath, allow_pickle=True)
                print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
            # Only examine first 3 files
            break
    else:
        continue
    break

# Also check 3D skeleton
skel_3d = os.path.join(MMASD_ROOT, '3D skeleton')
if os.path.exists(skel_3d):
    print(f"\n--- 3D skeleton contents ---")
    for item in os.listdir(skel_3d):
        path = os.path.join(skel_3d, item)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"  [DIR] {item} ({count} items)")
        else:
            size = os.path.getsize(path)
            print(f"  [FILE] {item} ({size/1e6:.1f} MB)")
    
    # Check for zip files in 3D
    for item in os.listdir(skel_3d):
        if item.endswith('.zip'):
            zpath = os.path.join(skel_3d, item)
            with zipfile.ZipFile(zpath, 'r') as zf:
                names = zf.namelist()
                print(f"\n--- ZIP: {item} ({len(names)} files) ---")
                for n in names[:15]:
                    print(f"  {n}")
                if len(names) > 15:
                    print(f"  ... and {len(names)-15} more")

print("\n" + "=" * 60)
print("Exploration complete!")
sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Output written to {OUT_FILE}")
