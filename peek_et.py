"""Explore Eye-Tracking dataset in detail."""
import os
import sys
import pandas as pd

OUT_FILE = r'C:\Users\ravik\AutismCare\autism_ai_prototype_v2\et_peek_output.txt'
sys.stdout = open(OUT_FILE, 'w', encoding='utf-8')

ET_ROOT = r'D:\Autism\AutismData\Eye-Tracking Dataset'

# Metadata
meta = pd.read_csv(os.path.join(ET_ROOT, 'Metadata_Participants.csv'))
print("=== METADATA ===")
print(f"Shape: {meta.shape}")
print(f"Columns: {meta.columns.tolist()}")
print(f"\nDistribution of 'Class':")
print(meta['Class'].value_counts().to_string())
print(f"\nFull metadata:")
print(meta.to_string())

# Eye-tracking output folder
et_out = os.path.join(ET_ROOT, 'Eye-tracking Output')
if os.path.exists(et_out):
    files = os.listdir(et_out)
    print(f"\n\n=== Eye-tracking Output folder ===")
    print(f"Files ({len(files)}):")
    for f in files:
        fpath = os.path.join(et_out, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")
    
    # Read sample CSV
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        sample = os.path.join(et_out, csv_files[0])
        # Try different separators
        for sep in ['\t', ',', ';']:
            try:
                df = pd.read_csv(sample, sep=sep, nrows=5, on_bad_lines='skip')
                if len(df.columns) > 3:
                    print(f"\n\nSample file: {csv_files[0]} (separator='{sep}')")
                    print(f"Shape: {df.shape}")
                    print(f"Columns ({len(df.columns)}):")
                    for i, c in enumerate(df.columns):
                        print(f"  [{i}] {c}")
                    print(f"\nFirst 3 rows:")
                    print(df.head(3).to_string())
                    break
            except:
                continue
        
        # Also check multiple files to see participant IDs
        print(f"\n\nAll CSV filenames:")
        for f in sorted(csv_files):
            print(f"  {f}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Done.")
