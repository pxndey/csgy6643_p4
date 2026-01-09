#!/usr/bin/env python3
"""
Inference script for GeoGuessr - Generate Kaggle submission
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# Import from training script
import sys
sys.path.append(str(Path(__file__).parent))
from train_geoguessr import GeoGuessrDataset, GeoLocatorModel

# Check for MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def generate_submission(
    model,
    test_loader,
    state_mapping: Dict[int, str],
    output_path: str
):
    """
    Generate Kaggle submission file
    
    Returns:
        DataFrame with columns:
        - sample_id
        - image_north, image_east, image_south, image_west
        - predicted_state_idx_1 through predicted_state_idx_5
        - predicted_latitude, predicted_longitude
    """
    model.eval()
    
    all_predictions = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sample_ids = batch['sample_id'].cpu().numpy()
            pixel_values = batch['pixel_values'].to(device)
            
            # Forward pass
            state_logits, gps_pred = model(pixel_values)
            
            # Get top-5 state predictions
            top5_probs, top5_indices = torch.topk(state_logits, k=5, dim=1)
            top5_indices = top5_indices.cpu().numpy()
            
            # Denormalize GPS predictions
            gps_pred = gps_pred.cpu().numpy()
            gps_pred[:, 0] = gps_pred[:, 0] * 13.0 + 37.0  # latitude
            gps_pred[:, 1] = gps_pred[:, 1] * 29.5 - 95.5  # longitude
            
            # Store predictions
            for i in range(len(sample_ids)):
                pred = {
                    'sample_id': sample_ids[i],
                    'predicted_state_idx_1': top5_indices[i, 0],
                    'predicted_state_idx_2': top5_indices[i, 1],
                    'predicted_state_idx_3': top5_indices[i, 2],
                    'predicted_state_idx_4': top5_indices[i, 3],
                    'predicted_state_idx_5': top5_indices[i, 4],
                    'predicted_latitude': gps_pred[i, 0],
                    'predicted_longitude': gps_pred[i, 1],
                }
                all_predictions.append(pred)
    
    # Create submission dataframe
    submission_df = pd.DataFrame(all_predictions)
    
    # Load sample submission to get image filenames
    sample_sub = pd.read_csv(Path(output_path).parent.parent / 'kaggle_dataset' / 'sample_submission.csv')
    
    # Merge with sample submission to get image columns
    submission_df = submission_df.merge(
        sample_sub[['sample_id', 'image_north', 'image_east', 'image_south', 'image_west']],
        on='sample_id',
        how='left'
    )
    
    # Reorder columns to match submission format
    submission_df = submission_df[[
        'sample_id', 'image_north', 'image_east', 'image_south', 'image_west',
        'predicted_state_idx_1', 'predicted_state_idx_2', 'predicted_state_idx_3',
        'predicted_state_idx_4', 'predicted_state_idx_5',
        'predicted_latitude', 'predicted_longitude'
    ]]
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Display sample predictions
    print("\nSample predictions:")
    print(submission_df.head(10))
    
    return submission_df


def main():
    # Configuration
    DATA_DIR = Path("/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset")
    TEST_CSV = DATA_DIR / "sample_submission.csv"
    TEST_IMG_DIR = DATA_DIR / "test_images"
    STATE_MAPPING_CSV = DATA_DIR / "state_mapping.csv"
    
    OUTPUT_DIR = Path("./outputs")
    MODEL_PATH = OUTPUT_DIR / "best_model.pth"  # or "latest_model.pth"
    SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
    
    BATCH_SIZE = 32  # Larger batch size for inference
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_geoguessr.py")
        return
        
    state_df = pd.read_csv(STATE_MAPPING_CSV)
    state_mapping = dict(zip(state_df['state_idx'], state_df['state']))

    # IMPORTANT: Use same num_states as training (50, not 33)
    num_states = 50  # Must match training
    print(f"Using num_states: {num_states}")
    
    # Load StreetCLIP processor (needed for image preprocessing)
    print("Loading StreetCLIP processor...")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    
    # Create model architecture
    model = GeoLocatorModel(
        clip_model=clip_model,
        num_states=num_states,
        freeze_backbone=True,
        use_4_views=True
    )
    
    # Load trained weights
    print(f"Loading model weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation score: {checkpoint['val_score']:.4f}")
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = GeoGuessrDataset(
        csv_path=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        processor=processor,
        state_mapping=state_mapping,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'mps' else False
    )
    
    # Generate submission
    submission_df = generate_submission(
        model=model,
        test_loader=test_loader,
        state_mapping=state_mapping,
        output_path=SUBMISSION_PATH
    )
    
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"Submission file ready for Kaggle upload: {SUBMISSION_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()