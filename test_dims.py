#!/usr/bin/env python3
"""
Quick test to verify StreetCLIP model dimensions
Run this before training to check everything works
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

print("Testing StreetCLIP model...")

# Load model
print("\n1. Loading StreetCLIP...")
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

print(f"   Model loaded successfully")

# Check dimensions
print("\n2. Checking model dimensions:")
print(f"   Vision config hidden size: {model.config.vision_config.hidden_size}")
print(f"   Projection dim: {model.config.projection_dim}")
print(f"   Text config hidden size: {model.config.text_config.hidden_size}")

# Test with dummy image
print("\n3. Testing with dummy image...")
dummy_image = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))

dummy_text = "a random street scene"
inputs = processor(images=dummy_image, text=dummy_text, return_tensors="pt")
print(f"   Input pixel values shape: {inputs['pixel_values'].shape}")

# Get vision features
with torch.no_grad():
    vision_outputs = model.vision_model(inputs['pixel_values'])
    image_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
    print(f"   CLS token (before projection) shape: {image_features.shape}")
    
    image_features_projected = model.visual_projection(image_features)
    print(f"   After visual projection shape: {image_features_projected.shape}")
    
    # Full model output
    outputs = model(**inputs)
    print(f"   Image embeddings shape: {outputs.image_embeds.shape}")

print("\n4. Expected dimension for model heads:")
print(f"   ✓ Use projection_dim = {model.config.projection_dim} for Linear layer input")

print("\n✅ Test complete! Model is ready to use.")
print(f"   Embedding dimension: {model.config.projection_dim}")