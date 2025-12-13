#!/usr/bin/env python3
"""
GeoGuessr Street View State & GPS Location Prediction
Using StreetCLIP for efficient fine-tuning on M4 Pro MacBook

Author: Pandey
Course: CS-GY 6643 Computer Vision - Project 4
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import wandb  # Optional: for experiment tracking

# Check for MPS (Metal Performance Shaders) on Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class GeoGuessrDataset(Dataset):
    """Dataset for loading 4-directional street view images with GPS coordinates"""
    
    def __init__(
        self, 
        csv_path: str, 
        img_dir: str, 
        processor: CLIPProcessor,
        state_mapping: Dict[int, str],
        is_test: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.state_mapping = state_mapping
        self.is_test = is_test
        
        # Create inverse mapping for state names to indices
        self.state_to_idx = {v: k for k, v in state_mapping.items()}
        
        print(f"Loaded {'test' if is_test else 'train'} dataset: {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load all 4 directional images
        directions = ['north', 'east', 'south', 'west']
        images = []
        
        for direction in directions:
            img_path = self.img_dir / row[f'image_{direction}']
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Create blank image as fallback
                images.append(Image.new('RGB', (256, 256), color=(128, 128, 128)))
        
        # Process images
        pixel_values = self.processor(images=images, return_tensors="pt")['pixel_values']
        
        if self.is_test:
            return {
                'sample_id': row['sample_id'],
                'pixel_values': pixel_values,  # Shape: [4, 3, 336, 336]
            }
        else:
            # Training data includes labels
            state_idx = int(row['state_idx'])
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            
            # Normalize GPS coordinates to [-1, 1] range for better training
            # US bounds approximately: lat [24, 50], lon [-125, -66]
            norm_lat = (latitude - 37.0) / 13.0  # Center ~37, range ~26
            norm_lon = (longitude + 95.5) / 29.5  # Center ~-95.5, range ~59
            
            return {
                'sample_id': row['sample_id'],
                'pixel_values': pixel_values,
                'state_idx': state_idx,
                'gps_coords': torch.tensor([norm_lat, norm_lon], dtype=torch.float32),
                'raw_coords': torch.tensor([latitude, longitude], dtype=torch.float32),
            }


class GeoLocatorModel(nn.Module):
    """
    Dual-task model for state classification and GPS regression
    Uses StreetCLIP as backbone with custom heads
    """
    
    def __init__(
        self, 
        clip_model: CLIPModel, 
        num_states: int = 50,
        freeze_backbone: bool = True,
        use_4_views: bool = True
    ):
        super().__init__()
        self.clip = clip_model
        self.use_4_views = use_4_views
        
        # Freeze CLIP backbone for efficient training
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False
            print("CLIP backbone frozen - only training custom heads")
        
        # Get CLIP vision embedding dimension
        # Use projection_dim for the final output dimension after visual projection
        if hasattr(self.clip.config, 'projection_dim'):
            self.embed_dim = self.clip.config.projection_dim
        else:
            # Fallback to vision hidden size if projection_dim not available
            self.embed_dim = self.clip.config.vision_config.hidden_size
        
        print(f"Using embedding dimension: {self.embed_dim}")
        
        # If using 4 views, we'll average them
        input_dim = self.embed_dim
        
        # State classification head with dropout for regularization
        self.state_classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_states)
        )
        
        # GPS regression head
        self.gps_regressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # latitude, longitude
        )
        
    def encode_images(self, pixel_values):
        """
        Encode images using CLIP vision encoder
        Args:
            pixel_values: [batch_size, 4, 3, 336, 336] for 4 directional views
        Returns:
            features: [batch_size, embed_dim]
        """
        batch_size = pixel_values.shape[0]
        num_views = pixel_values.shape[1]
        
        # Reshape to process all views at once
        # [batch_size, 4, 3, 336, 336] -> [batch_size*4, 3, 336, 336]
        pixel_values_flat = pixel_values.view(-1, *pixel_values.shape[2:])
        
        # Get CLIP image features
        with torch.no_grad() if not self.training else torch.enable_grad():
            vision_outputs = self.clip.vision_model(pixel_values_flat)
            image_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
            image_features = self.clip.visual_projection(image_features)
        
        # Reshape back and average across views
        # [batch_size*4, embed_dim] -> [batch_size, 4, embed_dim]
        image_features = image_features.view(batch_size, num_views, -1)
        
        # Average pooling across 4 directional views
        aggregated_features = image_features.mean(dim=1)  # [batch_size, embed_dim]
        
        return aggregated_features
    
    def forward(self, pixel_values):
        """
        Forward pass for both tasks
        Returns:
            state_logits: [batch_size, num_states]
            gps_coords: [batch_size, 2]
        """
        # Encode images
        features = self.encode_images(pixel_values)
        
        # Task 1: State classification
        state_logits = self.state_classifier(features)
        
        # Task 2: GPS regression
        gps_coords = self.gps_regressor(features)
        
        return state_logits, gps_coords


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth
    Returns distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    r = 6371.0
    
    return c * r


def compute_gps_score(distances_km):
    """
    Compute GPS score using normalized Haversine distance
    GPS Score = max(0, 1 - mean_distance_km / 5000)
    """
    mean_dist = np.mean(distances_km)
    score = max(0, 1 - mean_dist / 5000)
    return score


def compute_top_k_score(predictions, targets, k=5):
    """
    Compute weighted top-k scoring for state classification
    Weights: [1.00, 0.60, 0.40, 0.25, 0.15]
    """
    weights = np.array([1.00, 0.60, 0.40, 0.25, 0.15])
    batch_size = predictions.shape[0]
    
    # Get top-k predictions
    _, top_k_indices = torch.topk(predictions, k, dim=1)
    top_k_indices = top_k_indices.cpu().numpy()
    targets = targets.cpu().numpy()
    
    scores = []
    for i in range(batch_size):
        score = 0
        for rank, pred_idx in enumerate(top_k_indices[i]):
            if pred_idx == targets[i]:
                score = weights[rank]
                break
        scores.append(score)
    
    return np.array(scores)


class CombinedLoss(nn.Module):
    """
    Combined loss for dual-task learning
    Total Loss = α * Classification Loss + β * GPS Loss
    """
    
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, state_logits, gps_pred, state_target, gps_target):
        # Classification loss
        cls_loss = self.ce_loss(state_logits, state_target)
        
        # GPS regression loss (MSE on normalized coordinates)
        gps_loss = self.mse_loss(gps_pred, gps_target)
        
        # Combined loss
        total_loss = self.alpha * cls_loss + self.beta * gps_loss
        
        return total_loss, cls_loss, gps_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_gps_loss = 0
    all_state_preds = []
    all_state_targets = []
    all_gps_distances = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        state_targets = batch['state_idx'].to(device)
        gps_targets = batch['gps_coords'].to(device)
        raw_coords = batch['raw_coords'].cpu().numpy()
        
        # Forward pass
        optimizer.zero_grad()
        state_logits, gps_pred = model(pixel_values)
        
        # Compute loss
        loss, cls_loss, gps_loss = criterion(state_logits, gps_pred, state_targets, gps_targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_gps_loss += gps_loss.item()
        
        # Store predictions for evaluation
        all_state_preds.append(state_logits.detach())
        all_state_targets.append(state_targets.detach())
        
        # Denormalize GPS predictions and compute distances
        gps_pred_denorm = gps_pred.detach().cpu().numpy()
        gps_pred_denorm[:, 0] = gps_pred_denorm[:, 0] * 13.0 + 37.0  # lat
        gps_pred_denorm[:, 1] = gps_pred_denorm[:, 1] * 29.5 - 95.5  # lon
        
        for i in range(len(raw_coords)):
            dist = haversine_distance(
                raw_coords[i, 0], raw_coords[i, 1],
                gps_pred_denorm[i, 0], gps_pred_denorm[i, 1]
            )
            all_gps_distances.append(dist)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'gps': f'{gps_loss.item():.4f}'
        })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_gps_loss = total_gps_loss / len(dataloader)
    
    # Classification accuracy
    all_state_preds = torch.cat(all_state_preds, dim=0)
    all_state_targets = torch.cat(all_state_targets, dim=0)
    top1_acc = (all_state_preds.argmax(dim=1) == all_state_targets).float().mean().item()
    
    # Top-k score
    top_k_scores = compute_top_k_score(all_state_preds, all_state_targets, k=5)
    avg_top_k_score = np.mean(top_k_scores)
    
    # GPS score
    gps_score = compute_gps_score(all_gps_distances)
    median_dist = np.median(all_gps_distances)
    
    # Combined score (matches Kaggle evaluation)
    combined_score = 0.7 * avg_top_k_score + 0.3 * gps_score
    
    metrics = {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'gps_loss': avg_gps_loss,
        'top1_acc': top1_acc,
        'top_k_score': avg_top_k_score,
        'gps_score': gps_score,
        'median_dist_km': median_dist,
        'combined_score': combined_score
    }
    
    return metrics


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_gps_loss = 0
    all_state_preds = []
    all_state_targets = []
    all_gps_distances = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            state_targets = batch['state_idx'].to(device)
            gps_targets = batch['gps_coords'].to(device)
            raw_coords = batch['raw_coords'].cpu().numpy()
            
            # Forward pass
            state_logits, gps_pred = model(pixel_values)
            
            # Compute loss
            loss, cls_loss, gps_loss = criterion(state_logits, gps_pred, state_targets, gps_targets)
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_gps_loss += gps_loss.item()
            
            # Store predictions
            all_state_preds.append(state_logits)
            all_state_targets.append(state_targets)
            
            # Denormalize GPS predictions
            gps_pred_denorm = gps_pred.cpu().numpy()
            gps_pred_denorm[:, 0] = gps_pred_denorm[:, 0] * 13.0 + 37.0
            gps_pred_denorm[:, 1] = gps_pred_denorm[:, 1] * 29.5 - 95.5
            
            for i in range(len(raw_coords)):
                dist = haversine_distance(
                    raw_coords[i, 0], raw_coords[i, 1],
                    gps_pred_denorm[i, 0], gps_pred_denorm[i, 1]
                )
                all_gps_distances.append(dist)
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_gps_loss = total_gps_loss / len(dataloader)
    
    all_state_preds = torch.cat(all_state_preds, dim=0)
    all_state_targets = torch.cat(all_state_targets, dim=0)
    top1_acc = (all_state_preds.argmax(dim=1) == all_state_targets).float().mean().item()
    
    top_k_scores = compute_top_k_score(all_state_preds, all_state_targets, k=5)
    avg_top_k_score = np.mean(top_k_scores)
    
    gps_score = compute_gps_score(all_gps_distances)
    median_dist = np.median(all_gps_distances)
    
    combined_score = 0.7 * avg_top_k_score + 0.3 * gps_score
    
    metrics = {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'gps_loss': avg_gps_loss,
        'top1_acc': top1_acc,
        'top_k_score': avg_top_k_score,
        'gps_score': gps_score,
        'median_dist_km': median_dist,
        'combined_score': combined_score
    }
    
    return metrics


def main():
    # Configuration
    DATA_DIR = Path("/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset")
    TRAIN_CSV = DATA_DIR / "train_ground_truth.csv"
    TRAIN_IMG_DIR = DATA_DIR / "train_images"
    STATE_MAPPING_CSV = DATA_DIR / "state_mapping.csv"
    
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32  # Adjust based on memory
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2
    
    # Load state mapping
    state_df = pd.read_csv(STATE_MAPPING_CSV)
    state_mapping = dict(zip(state_df['state_idx'], state_df['state']))
    num_states = len(state_mapping)
    print(f"Number of states: {num_states}")
    
    # Load StreetCLIP model and processor
    print("Loading StreetCLIP model...")
    clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    
    # Create model
    model = GeoLocatorModel(
        clip_model=clip_model,
        num_states=num_states,
        freeze_backbone=True,  # Freeze CLIP for efficient training
        use_4_views=True
    )
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Create datasets
    print("Loading datasets...")
    full_dataset = GeoGuessrDataset(
        csv_path=TRAIN_CSV,
        img_dir=TRAIN_IMG_DIR,
        processor=processor,
        state_mapping=state_mapping,
        is_test=False
    )
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'mps' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'mps' else False
    )
    
    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.7, beta=0.3)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_val_score = 0.0
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Top-1 Acc: {train_metrics['top1_acc']:.4f} | "
              f"Top-K Score: {train_metrics['top_k_score']:.4f} | "
              f"GPS Score: {train_metrics['gps_score']:.4f} | "
              f"Combined: {train_metrics['combined_score']:.4f} | "
              f"Median Dist: {train_metrics['median_dist_km']:.1f} km")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
              f"Top-1 Acc: {val_metrics['top1_acc']:.4f} | "
              f"Top-K Score: {val_metrics['top_k_score']:.4f} | "
              f"GPS Score: {val_metrics['gps_score']:.4f} | "
              f"Combined: {val_metrics['combined_score']:.4f} | "
              f"Median Dist: {val_metrics['median_dist_km']:.1f} km")
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_metrics['combined_score'] > best_val_score:
            best_val_score = val_metrics['combined_score']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': best_val_score,
                'metrics': val_metrics
            }
            torch.save(checkpoint, OUTPUT_DIR / 'best_model.pth')
            print(f"✓ Saved best model (score: {best_val_score:.4f})")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_score': val_metrics['combined_score'],
            'metrics': val_metrics
        }
        torch.save(checkpoint, OUTPUT_DIR / 'latest_model.pth')
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation score: {best_val_score:.4f} at epoch {best_epoch}")
    print(f"Models saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()