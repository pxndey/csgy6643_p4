"""
Configuration file for GeoGuessr Training
Modify these parameters to experiment with different settings
"""

import torch

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = "/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset"
OUTPUT_DIR = "./outputs"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "geolocal/StreetCLIP"  # HuggingFace model identifier
FREEZE_BACKBONE = True  # If True, only train classification and GPS heads
                        # If False, fine-tune entire CLIP model (requires more memory)

# Number of states to predict (auto-detected from state_mapping.csv)
NUM_STATES = None  # Set to None for auto-detection

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Batch size (reduce if you get out-of-memory errors)
BATCH_SIZE = 16  # Options: 4, 8, 12, 16, 20, 24, 32
                 # M4 Pro 24GB: safe range is 8-20

# Number of training epochs
NUM_EPOCHS = 15  # Typical range: 10-25

# Learning rate
LEARNING_RATE = 1e-4  # Options: 5e-5, 1e-4, 2e-4, 3e-4

# Weight decay (L2 regularization)
WEIGHT_DECAY = 1e-4  # Options: 0, 1e-5, 1e-4, 1e-3

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"  # Options: "cosine", "step", "plateau"
MIN_LR = 1e-6  # Minimum learning rate for cosine scheduler

# Warmup epochs (gradual learning rate increase at start)
WARMUP_EPOCHS = 2

# Gradient clipping (prevents exploding gradients)
GRADIENT_CLIP_NORM = 1.0

# =============================================================================
# LOSS CONFIGURATION
# =============================================================================
# Loss weights for dual-task learning
# These should match Kaggle evaluation weights
CLASSIFICATION_WEIGHT = 0.7  # Weight for state classification loss
GPS_WEIGHT = 0.3             # Weight for GPS regression loss

# Note: CLASSIFICATION_WEIGHT + GPS_WEIGHT should equal 1.0 for proper scaling

# =============================================================================
# DATA AUGMENTATION (Optional - currently not implemented)
# =============================================================================
USE_AUGMENTATION = False
AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,    # Probability of horizontal flip
    'rotation': 5,             # Max rotation degrees
    'color_jitter': 0.1,       # Color jitter strength
    'random_crop': False,      # Whether to use random crops
}

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Dropout rates for regularization
DROPOUT_RATE_1 = 0.3  # Dropout after first hidden layer
DROPOUT_RATE_2 = 0.2  # Dropout after second hidden layer

# Hidden layer dimensions
STATE_CLASSIFIER_HIDDEN = [512, 256]  # [layer1_dim, layer2_dim]
GPS_REGRESSOR_HIDDEN = [512, 256, 128]  # [layer1_dim, layer2_dim, layer3_dim]

# Use all 4 directional views (North, East, South, West)
USE_4_VIEWS = True  # If False, only use North view (not recommended)

# =============================================================================
# DATA LOADING
# =============================================================================
# Number of workers for data loading (adjust based on CPU cores)
NUM_WORKERS = 4  # Options: 0, 2, 4, 6, 8
                 # Use 0 for debugging (single-threaded)
                 # Use 4-8 for M4 Pro (8-12 cores)

# Pin memory for faster GPU transfer (only useful for CUDA, not MPS)
PIN_MEMORY = False  # Keep False for MPS (Mac)

# Train/validation split ratio
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation
RANDOM_SEED = 42       # For reproducible splits

# =============================================================================
# GPS COORDINATE NORMALIZATION
# =============================================================================
# US bounds for GPS normalization (approximate)
# These are used to normalize coordinates to [-1, 1] range for better training

GPS_LAT_CENTER = 37.0   # Center of US latitude range
GPS_LAT_RANGE = 13.0    # Half-range for latitude (covers ~24°N to 50°N)

GPS_LON_CENTER = -95.5  # Center of US longitude range
GPS_LON_RANGE = 29.5    # Half-range for longitude (covers ~-125°W to -66°W)

# =============================================================================
# CHECKPOINTING
# =============================================================================
# Save model checkpoint every N epochs
SAVE_EVERY_N_EPOCHS = 5

# Keep best model based on this metric
BEST_METRIC = "combined_score"  # Options: "combined_score", "top_k_score", "gps_score", "val_loss"

# Save both best and latest checkpoints
SAVE_BEST = True
SAVE_LATEST = True

# =============================================================================
# LOGGING & EXPERIMENT TRACKING
# =============================================================================
# Use Weights & Biases for experiment tracking
USE_WANDB = False  # Set to True if you want to use W&B
WANDB_PROJECT = "geoguessr-streetclip"
WANDB_ENTITY = None  # Your W&B username (None for default)

# Print training progress
PRINT_EVERY_N_BATCHES = 100  # Print metrics every N batches

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
# Batch size for inference (can be larger than training)
INFERENCE_BATCH_SIZE = 32  # Options: 16, 32, 64

# Which checkpoint to use for inference
INFERENCE_CHECKPOINT = "best_model.pth"  # Options: "best_model.pth", "latest_model.pth"

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
# Device selection (auto-detected, but can be overridden)
# Options: "auto", "mps", "cuda", "cpu"
DEVICE = "auto"

# Mixed precision training (experimental for MPS)
USE_AMP = False  # Automatic Mixed Precision (FP16)
                 # Note: AMP support on MPS is limited, keep False for now

# =============================================================================
# ADVANCED OPTIONS
# =============================================================================
# Gradient accumulation (simulates larger batch size)
GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch size = BATCH_SIZE × this value
                                  # Increase if you want larger effective batch but have limited memory

# Early stopping
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for N epochs
EARLY_STOPPING_METRIC = "combined_score"

# Resume training from checkpoint
RESUME_FROM_CHECKPOINT = None  # Path to checkpoint, or None to start fresh

# =============================================================================
# EVALUATION METRICS
# =============================================================================
# Top-K weights for state classification evaluation
TOP_K_WEIGHTS = [1.00, 0.60, 0.40, 0.25, 0.15]  # Matches Kaggle evaluation

# GPS score normalization factor (in kilometers)
GPS_NORMALIZATION_FACTOR = 5000  # GPS score = max(0, 1 - distance/5000)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_device():
    """Auto-detect the best available device"""
    if DEVICE != "auto":
        return torch.device(DEVICE)
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def normalize_gps_coords(lat, lon):
    """
    Normalize GPS coordinates to [-1, 1] range
    
    Args:
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees
    
    Returns:
        norm_lat: Normalized latitude
        norm_lon: Normalized longitude
    """
    norm_lat = (lat - GPS_LAT_CENTER) / GPS_LAT_RANGE
    norm_lon = (lon - GPS_LON_CENTER) / GPS_LON_RANGE
    return norm_lat, norm_lon


def denormalize_gps_coords(norm_lat, norm_lon):
    """
    Denormalize GPS coordinates back to degrees
    
    Args:
        norm_lat: Normalized latitude(s)
        norm_lon: Normalized longitude(s)
    
    Returns:
        lat: Latitude in degrees
        lon: Longitude in degrees
    """
    lat = norm_lat * GPS_LAT_RANGE + GPS_LAT_CENTER
    lon = norm_lon * GPS_LON_RANGE + GPS_LON_CENTER
    return lat, lon


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Freeze backbone: {FREEZE_BACKBONE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {get_device()}")
    print(f"Classification weight: {CLASSIFICATION_WEIGHT}")
    print(f"GPS weight: {GPS_WEIGHT}")
    print(f"Dropout rates: {DROPOUT_RATE_1}, {DROPOUT_RATE_2}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Train/val split: {TRAIN_VAL_SPLIT:.2%}")
    print("=" * 80)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def use_fast_config():
    """Quick training config for testing (lower quality)"""
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-4
    print("Using FAST config (for testing only)")


def use_best_config():
    """Best known configuration (slower but higher quality)"""
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    BATCH_SIZE = 12
    NUM_EPOCHS = 25
    LEARNING_RATE = 8e-5
    WEIGHT_DECAY = 5e-5
    print("Using BEST config (recommended for competition)")


def use_memory_efficient_config():
    """Memory-efficient config for machines with limited RAM"""
    global BATCH_SIZE, NUM_WORKERS, GRADIENT_ACCUMULATION_STEPS
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16
    print("Using MEMORY_EFFICIENT config")


if __name__ == "__main__":
    # Print current configuration
    print_config()