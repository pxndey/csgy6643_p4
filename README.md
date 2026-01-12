# GeoGuessr Street View State & GPS Location Prediction

A dual-task computer vision model that predicts both US state classifications and precise GPS coordinates from Google Street View images using StreetCLIP for efficient fine-tuning.

## Overview

This project implements a geolocation prediction system that processes 4-directional street view images (North, East, South, West) to predict:
1. **State Classification**: Top-5 weighted predictions of US states
2. **GPS Coordinates**: Precise latitude/longitude prediction

The model uses StreetCLIP (a geolocation-specialized CLIP model) as a frozen backbone with custom task-specific heads, optimized for training on M4 Pro MacBook with MPS acceleration.

## Features

- **Multi-view Processing**: Aggregates features from 4 directional street view images
- **Dual-task Learning**: Simultaneous state classification and GPS regression
- **Efficient Training**: Frozen backbone with only custom heads trainable (~1-2% of total parameters)
- **Custom Scoring Metrics**: 
  - Weighted top-5 classification scoring (1.00, 0.60, 0.40, 0.25, 0.15)
  - GPS scoring using normalized Haversine distance
  - Combined score: 0.7 × state_score + 0.3 × gps_score
- **MPS Acceleration**: Native Apple Silicon GPU support

## Architecture
```
Street View Images (4 directions)
         ↓
    StreetCLIP Encoder (frozen)
         ↓
    Average Pooling
         ↓
    ┌──────────────┴──────────────┐
    ↓                             ↓
State Classifier              GPS Regressor
(3-layer MLP)                (4-layer MLP)
    ↓                             ↓
50 State Logits           [Latitude, Longitude]
```

## Requirements
```bash
torch>=2.0.0
transformers>=4.30.0
pandas
numpy
Pillow
tqdm
wandb  # optional
```

## Dataset Structure
```
kaggle_dataset/
├── train_ground_truth.csv
├── state_mapping.csv
└── train_images/
    ├── [sample_id]_north.jpg
    ├── [sample_id]_east.jpg
    ├── [sample_id]_south.jpg
    └── [sample_id]_west.jpg
```

### CSV Format

**train_ground_truth.csv**:
- `sample_id`: Unique identifier
- `state_idx`: State index (0-49)
- `latitude`: GPS latitude
- `longitude`: GPS longitude
- `image_north`, `image_east`, `image_south`, `image_west`: Image filenames

**state_mapping.csv**:
- `state_idx`: Index
- `state`: State abbreviation

## Usage

### Training
```bash
python train_geoguesser.py
```

### Configuration

Key hyperparameters in `main()`:
```python
BATCH_SIZE = 32          # Adjust based on available memory
NUM_EPOCHS = 15          # Training epochs
LEARNING_RATE = 1e-4     # AdamW learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization
```

Loss function weights:
```python
criterion = CombinedLoss(alpha=0.7, beta=0.3)
# alpha: state classification loss weight
# beta: GPS regression loss weight
```

## Model Details

### GeoLocatorModel

- **Backbone**: StreetCLIP (frozen)
- **Input**: 4 directional images [batch_size, 4, 3, 336, 336]
- **Feature Aggregation**: Average pooling across 4 views
- **State Classifier**: 3-layer MLP (512→256→50) with LayerNorm and Dropout
- **GPS Regressor**: 4-layer MLP (512→256→128→2) with LayerNorm and Dropout

### Loss Function
```python
Total_Loss = 0.7 × CrossEntropy(state_pred, state_true) 
           + 0.3 × MSE(gps_pred_normalized, gps_true_normalized)
```

GPS coordinates are normalized to [-1, 1] range:
- Latitude: (lat - 37.0) / 13.0
- Longitude: (lon + 95.5) / 29.5

## Evaluation Metrics

### State Classification Score
Weighted top-5 scoring:
- 1st place: 1.00
- 2nd place: 0.60
- 3rd place: 0.40
- 4th place: 0.25
- 5th place: 0.15

### GPS Score
```python
GPS_Score = max(0, 1 - mean_haversine_distance_km / 5000)
```

### Combined Score
```python
Final_Score = 0.7 × State_Score + 0.3 × GPS_Score
```

## Training Output

The model saves:
- `outputs/best_model.pth`: Best validation checkpoint
- `outputs/latest_model.pth`: Most recent checkpoint

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Validation score
- Full metrics dictionary

## Performance Tracking

Training logs include:
- Total loss, classification loss, GPS loss
- Top-1 accuracy
- Top-5 weighted score
- GPS score
- Median Haversine distance (km)
- Combined score

## Technical Notes

### Device Support
- **MPS (Apple Silicon)**: Primary target, uses Metal Performance Shaders
- **CPU**: Fallback for non-MPS systems
- **CUDA**: Not explicitly tested but should work with minor modifications

### Memory Optimization
- Frozen StreetCLIP backbone reduces memory footprint
- Gradient checkpointing not needed due to frozen backbone
- Typical memory usage: ~4-6GB on M4 Pro with batch_size=32

### Data Normalization
GPS coordinates normalized to US bounds:
- Latitude range: [24°N, 50°N] → centered at 37°N
- Longitude range: [125°W, 66°W] → centered at 95.5°W

## Course Information

**Course**: CS-GY 6643 Computer Vision - Project 4  
**Author**: Pandey  
**Institution**: NYU Tandon School of Engineering

## License

Academic project - NYU Tandon School of Engineering

## Acknowledgments

- StreetCLIP model: [geolocal/StreetCLIP](https://huggingface.co/geolocal/StreetCLIP)
- Built with PyTorch and Hugging Face Transformers
