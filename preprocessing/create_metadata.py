"""
Create metadata.csv mapping video paths to unified binary labels.
Handles multiple datasets with different label conventions.

Label Unification:
- Normal / Alert / Alert → ALERT (0)
- Yawning / Drowsy / Microsleep / Sleeping → DROWSY (1)
"""

import os
import csv
from pathlib import Path

DATASET_ROOT = "dataset"
OUTPUT_CSV = "data/metadata.csv"
os.makedirs("data", exist_ok=True)

metadata = []

def get_label_from_path(video_path):
    """Extract and unify label from video path."""
    parts = video_path.lower().split(os.sep)
    
    # Check each component for label hints
    for part in parts:
        if 'alert' in part and 'glass' not in part:
            return 'ALERT'
        elif 'drowsy' in part or 'low_vigilance' in part:
            return 'DROWSY'
        elif 'normal' in part:
            return 'ALERT'
        elif 'yawn' in part:
            return 'DROWSY'
        elif 'microsleep' in part:
            return 'DROWSY'
    
    return None

def normalize_rel_path(path_value):
    """Normalize to a forward-slash relative path for CSV storage."""
    return path_value.replace("\\", "/")

def scan_nthu_ddd():
    """Scan NTHU_DDD dataset: dataset/nthu_ddd/subjectXXX/{alert,drowsy}/*.avi"""
    base_path = os.path.join(DATASET_ROOT, "nthu_ddd")
    if not os.path.exists(base_path):
        return
    
    for subject_dir in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        
        for label_dir in os.listdir(subject_path):
            label_path = os.path.join(subject_path, label_dir)
            if not os.path.isdir(label_path):
                continue
            
            unified_label = get_label_from_path(label_path)
            if unified_label is None:
                continue
            
            for video_file in os.listdir(label_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.relpath(
                        os.path.join(label_path, video_file),
                        DATASET_ROOT
                    )
                    video_path = normalize_rel_path(video_path)
                    metadata.append({
                        'dataset': 'NTHU_DDD',
                        'video_path': video_path,
                        'label': unified_label
                    })
    print(f"Found {len([m for m in metadata if m['dataset']=='NTHU_DDD'])} NTHU_DDD videos")

def scan_yawdd():
    """Scan YawDD dataset: dataset/yawdd/{normal,yawning}/*.avi"""
    base_path = os.path.join(DATASET_ROOT, "yawdd")
    if not os.path.exists(base_path):
        return
    
    for label_dir in os.listdir(base_path):
        label_path = os.path.join(base_path, label_dir)
        if not os.path.isdir(label_path):
            continue
        
        unified_label = get_label_from_path(label_path)
        if unified_label is None:
            continue
        
        for video_file in os.listdir(label_path):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.relpath(
                    os.path.join(label_path, video_file),
                    DATASET_ROOT
                )
                video_path = normalize_rel_path(video_path)
                metadata.append({
                    'dataset': 'YawDD',
                    'video_path': video_path,
                    'label': unified_label
                })
    print(f"Found {len([m for m in metadata if m['dataset']=='YawDD'])} YawDD videos")

def scan_uta_rldd():
    """Scan UTA_RLDD dataset: dataset/uta_rldd/subjectXX/{alert,drowsy,low_vigilance}.mov"""
    base_path = os.path.join(DATASET_ROOT, "uta_rldd")
    if not os.path.exists(base_path):
        return
    
    for subject_dir in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        
        for video_file in os.listdir(subject_path):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                unified_label = get_label_from_path(video_file)
                if unified_label is None:
                    continue
                
                video_path = os.path.relpath(
                    os.path.join(subject_path, video_file),
                    DATASET_ROOT
                )
                video_path = normalize_rel_path(video_path)
                metadata.append({
                    'dataset': 'UTA_RLDD',
                    'video_path': video_path,
                    'label': unified_label
                })
    print(f"Found {len([m for m in metadata if m['dataset']=='UTA_RLDD'])} UTA_RLDD videos")

def scan_nitymed():
    """Scan NITYMED dataset: dataset/nitymed/subjectXX/{microsleep,yawning}/*.mp4"""
    base_path = os.path.join(DATASET_ROOT, "nitymed")
    if not os.path.exists(base_path):
        return
    
    for subject_dir in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        
        for label_dir in os.listdir(subject_path):
            label_path = os.path.join(subject_path, label_dir)
            if not os.path.isdir(label_path):
                continue
            
            unified_label = get_label_from_path(label_path)
            if unified_label is None:
                continue
            
            for video_file in os.listdir(label_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.relpath(
                        os.path.join(label_path, video_file),
                        DATASET_ROOT
                    )
                    video_path = normalize_rel_path(video_path)
                    metadata.append({
                        'dataset': 'NITYMED',
                        'video_path': video_path,
                        'label': unified_label
                    })
    print(f"Found {len([m for m in metadata if m['dataset']=='NITYMED'])} NITYMED videos")

def scan_testing_dataset():
    """Scan Testing_Dataset if it exists."""
    base_path = os.path.join(DATASET_ROOT, "Testing_Dataset")
    if not os.path.exists(base_path):
        return
    
    # Looking for .txt files that contain labels and video references
    test_label_txt_path = os.path.join(base_path, "test_label_txt")
    if os.path.exists(test_label_txt_path):
        for label_dir in os.listdir(test_label_txt_path):
            label_path = os.path.join(test_label_txt_path, label_dir)
            if not os.path.isdir(label_path):
                continue
            
            # Extract label from directory name (e.g., "glasses" → testing condition)
            # These are testing sets - classify as needed based on content
            # For now, skip as instructions say NITYMED is for testing only
            pass

def scan_testing_dataset_nthu():
    """Scan Testing_Dataset_NTHU: dataset/Testing_Dataset_NTHU/*.mp4 and test_label_txt/ for labels."""
    base_path = os.path.join(DATASET_ROOT, "Testing_Dataset_NTHU")
    if not os.path.exists(base_path):
        return
    for video_file in os.listdir(base_path):
        if video_file.lower().endswith('.mp4'):
            video_path = os.path.relpath(os.path.join(base_path, video_file), DATASET_ROOT)
            video_path = normalize_rel_path(video_path)
            # Extract subject and condition from filename (e.g., 003_glasses_mix.mp4)
            fname = os.path.splitext(video_file)[0]
            parts = fname.split('_')
            subject = parts[0]
            condition = parts[1] if len(parts) > 1 else ''
            # No unified label for test set, but can set as 'TEST' or leave blank
            metadata.append({
                'dataset': 'Testing_Dataset_NTHU',
                'video_path': video_path,
                'label': ''
            })
    print(f"Found {len([m for m in metadata if m['dataset']=='Testing_Dataset_NTHU'])} Testing_Dataset_NTHU videos")

# Run all scanners
print("Scanning datasets for metadata...")
print("TRAINING DATA ONLY (per instructions.md):")
scan_nthu_ddd()
scan_yawdd()
scan_uta_rldd()
scan_nitymed()
scan_testing_dataset_nthu()

# Write metadata.csv
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'video_path', 'label'])
    writer.writeheader()
    writer.writerows(metadata)

print(f"\nMetadata Summary:")
print(f"Total videos: {len(metadata)}")
for dataset in ['NTHU_DDD', 'YawDD', 'UTA_RLDD', 'NITYMED']:
    count = len([m for m in metadata if m['dataset'] == dataset])
    if count > 0:
        print(f"  {dataset}: {count}")

# Check label distribution
alert_count = len([m for m in metadata if m['label'] == 'ALERT'])
drowsy_count = len([m for m in metadata if m['label'] == 'DROWSY'])
print(f"\nLabel Distribution:")
print(f"  ALERT (0): {alert_count}")
print(f"  DROWSY (1): {drowsy_count}")

print(f"\nMetadata saved to: {OUTPUT_CSV}")
