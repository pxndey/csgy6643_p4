#!/usr/bin/env python3
"""
Profile the data loading pipeline to find bottlenecks
"""

import time
import torch
from pathlib import Path
from transformers import CLIPProcessor
from train_geoguessr import GeoGuessrDataset
import pandas as pd

DATA_DIR = Path("/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset")
TRAIN_CSV = DATA_DIR / "train_ground_truth.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"
STATE_MAPPING_CSV = DATA_DIR / "state_mapping.csv"

print("Profiling data loading pipeline...")
print("=" * 60)

# Load state mapping
state_df = pd.read_csv(STATE_MAPPING_CSV)
state_mapping = dict(zip(state_df['state_idx'], state_df['state']))

# Load processor
print("\n1. Loading CLIP processor...")
start = time.time()
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
print(f"   Time: {time.time() - start:.2f}s")

# Create dataset
print("\n2. Creating dataset...")
start = time.time()
dataset = GeoGuessrDataset(
    csv_path=TRAIN_CSV,
    img_dir=TRAIN_IMG_DIR,
    processor=processor,
    state_mapping=state_mapping,
    is_test=False
)
print(f"   Time: {time.time() - start:.2f}s")

# Test single sample loading
print("\n3. Loading single sample (4 images)...")
times = []
for i in range(10):
    start = time.time()
    sample = dataset[i]
    times.append(time.time() - start)
    if i == 0:
        print(f"   Sample 0 time: {times[0]:.3f}s")
        print(f"   Pixel values shape: {sample['pixel_values'].shape}")

avg_time = sum(times) / len(times)
print(f"   Average time per sample: {avg_time:.3f}s")
print(f"   Estimated time per batch (16 samples): {avg_time * 16:.2f}s")

# Test batch loading with DataLoader
print("\n4. Testing DataLoader (3 batches)...")
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
batch_times = []

for i, batch in enumerate(loader):
    if i >= 3:
        break
    start = time.time()
    # Simulate minimal processing
    _ = batch['pixel_values'].shape
    batch_times.append(time.time() - start)
    print(f"   Batch {i}: {batch_times[-1]:.3f}s")

print(f"\n5. Analysis:")
print(f"   Average sample load time: {avg_time:.3f}s")
print(f"   Expected batch time (16 samples): {avg_time * 16:.2f}s")
print(f"   With 3712 batches per epoch: {(avg_time * 16 * 3712) / 3600:.2f} hours")

# Check disk speed
print("\n6. Testing raw image load speed (no preprocessing)...")
from PIL import Image
import numpy as np

train_df = pd.read_csv(TRAIN_CSV)
img_times = []

for i in range(40):  # Test 40 images (10 samples × 4 directions)
    row = train_df.iloc[i // 4]
    direction = ['north', 'east', 'south', 'west'][i % 4]
    img_path = TRAIN_IMG_DIR / row[f'image_{direction}']
    
    start = time.time()
    img = Image.open(img_path).convert('RGB')
    _ = np.array(img)
    img_times.append(time.time() - start)

avg_img_time = sum(img_times) / len(img_times)
print(f"   Average raw image load: {avg_img_time*1000:.1f}ms")
print(f"   4 images per sample: {avg_img_time*4*1000:.1f}ms")

# Check preprocessing speed
print("\n7. Testing CLIP preprocessing speed...")
img = Image.open(TRAIN_IMG_DIR / train_df.iloc[0]['image_north']).convert('RGB')
images = [img] * 4

preprocess_times = []
for _ in range(10):
    start = time.time()
    _ = processor(images=images, return_tensors="pt")
    preprocess_times.append(time.time() - start)

avg_preprocess = sum(preprocess_times) / len(preprocess_times)
print(f"   Average preprocessing (4 images): {avg_preprocess*1000:.1f}ms")

print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS:")
print(f"   Image loading (4 imgs):    {avg_img_time*4*1000:.1f}ms")
print(f"   CLIP preprocessing:        {avg_preprocess*1000:.1f}ms")
print(f"   Total per sample:          {(avg_img_time*4 + avg_preprocess)*1000:.1f}ms")
print(f"   Per batch (16 samples):    {(avg_img_time*4 + avg_preprocess)*16:.2f}s")
print("=" * 60)

if avg_time > 0.5:
    print("\n⚠️  SLOW DATA LOADING DETECTED!")
    print("Possible causes:")
    print("  1. Slow disk I/O (SSD vs HDD)")
    print("  2. CLIP preprocessing overhead")
    print("  3. Large image sizes")
    print("\nRecommended solutions:")
    print("  - Preprocess and cache images")
    print("  - Use image compression")
    print("  - Load images to RAM if possible")