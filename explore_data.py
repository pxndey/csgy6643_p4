#!/usr/bin/env python3
"""
Data Exploration & Visualization Script
Quick analysis of the GeoGuessr dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def explore_dataset():
    """Explore and visualize the GeoGuessr dataset"""
    
    DATA_DIR = Path("/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset")
    
    print("=" * 80)
    print("GEOGUESSR DATASET EXPLORATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data files...")
    train_df = pd.read_csv(DATA_DIR / "train_ground_truth.csv")
    state_df = pd.read_csv(DATA_DIR / "state_mapping.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Test samples: {len(sample_sub):,}")
    print(f"   Number of states: {len(state_df)}")
    
    # Display sample data
    print("\n2. Sample training data:")
    print(train_df.head())
    
    print("\n3. State mapping:")
    print(state_df.head(10))
    
    # State distribution
    print("\n4. State distribution in training data:")
    state_counts = train_df['state'].value_counts()
    print(f"\n   Most common states:")
    for state, count in state_counts.head(10).items():
        print(f"   {state:20s}: {count:5d} samples ({100*count/len(train_df):.2f}%)")
    
    print(f"\n   Least common states:")
    for state, count in state_counts.tail(5).items():
        print(f"   {state:20s}: {count:5d} samples ({100*count/len(train_df):.2f}%)")
    
    # GPS statistics
    print("\n5. GPS coordinate statistics:")
    print(f"   Latitude range:  [{train_df['latitude'].min():.2f}, {train_df['latitude'].max():.2f}]")
    print(f"   Longitude range: [{train_df['longitude'].min():.2f}, {train_df['longitude'].max():.2f}]")
    print(f"   Latitude mean:   {train_df['latitude'].mean():.2f}")
    print(f"   Longitude mean:  {train_df['longitude'].mean():.2f}")
    
    # Check for missing values
    print("\n6. Missing values check:")
    print(train_df.isnull().sum())
    
    # Check image files exist
    print("\n7. Checking image files...")
    train_img_dir = DATA_DIR / "train_images"
    test_img_dir = DATA_DIR / "test_images"
    
    if train_img_dir.exists():
        train_images = list(train_img_dir.glob("*.jpg"))
        print(f"   Training images found: {len(train_images):,}")
        print(f"   Expected: {len(train_df) * 4:,} (4 per sample)")
    
    if test_img_dir.exists():
        test_images = list(test_img_dir.glob("*.jpg"))
        print(f"   Test images found: {len(test_images):,}")
        print(f"   Expected: {len(sample_sub) * 4:,} (4 per sample)")
    
    # Create visualizations
    print("\n8. Creating visualizations...")
    create_visualizations(train_df, state_counts)
    
    print("\n" + "=" * 80)
    print("Exploration complete! Check 'outputs/exploration_*.png' for plots.")
    print("=" * 80)


def create_visualizations(train_df, state_counts):
    """Create and save exploratory visualizations"""
    
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # 1. State distribution bar plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 20 states
    ax1 = axes[0, 0]
    state_counts.head(20).plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('Top 20 States by Sample Count', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('State')
    ax1.invert_yaxis()
    
    # GPS scatter plot
    ax2 = axes[0, 1]
    sample_df = train_df.sample(min(5000, len(train_df)), random_state=42)
    scatter = ax2.scatter(
        sample_df['longitude'], 
        sample_df['latitude'],
        c=sample_df['state_idx'],
        cmap='tab20',
        alpha=0.6,
        s=10
    )
    ax2.set_title('Geographic Distribution of Training Samples', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_xlim(-130, -60)
    ax2.set_ylim(20, 55)
    
    # Latitude distribution
    ax3 = axes[1, 0]
    ax3.hist(train_df['latitude'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax3.set_title('Latitude Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Latitude (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(train_df['latitude'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {train_df["latitude"].mean():.2f}')
    ax3.legend()
    
    # Longitude distribution
    ax4 = axes[1, 1]
    ax4.hist(train_df['longitude'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.set_title('Longitude Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Longitude (degrees)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(train_df['longitude'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {train_df["longitude"].mean():.2f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exploration_overview.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR / 'exploration_overview.png'}")
    plt.close()
    
    # 2. State imbalance plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_counts = state_counts.sort_values(ascending=False)
    ax.bar(range(len(sorted_counts)), sorted_counts.values, color='mediumseagreen')
    ax.set_title('Class Imbalance: All States Sorted by Frequency', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('State (sorted by frequency)')
    ax.set_ylabel('Number of Samples')
    ax.axhline(sorted_counts.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {sorted_counts.mean():.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exploration_class_imbalance.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR / 'exploration_class_imbalance.png'}")
    plt.close()
    
    # 3. Geographic heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create 2D histogram (heatmap)
    lat_bins = np.linspace(train_df['latitude'].min(), train_df['latitude'].max(), 50)
    lon_bins = np.linspace(train_df['longitude'].min(), train_df['longitude'].max(), 80)
    
    heatmap, xedges, yedges = np.histogram2d(
        train_df['longitude'], 
        train_df['latitude'],
        bins=[lon_bins, lat_bins]
    )
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                   cmap='YlOrRd', aspect='auto', interpolation='gaussian')
    
    ax.set_title('Geographic Density Heatmap of Training Samples', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sample Density', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exploration_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR / 'exploration_heatmap.png'}")
    plt.close()


def analyze_state_characteristics():
    """Analyze geographic characteristics of each state"""
    
    DATA_DIR = Path("/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset")
    train_df = pd.read_csv(DATA_DIR / "train_ground_truth.csv")
    
    print("\n9. State geographic characteristics:")
    print(f"   {'State':<20} {'Samples':<10} {'Lat Range':<15} {'Lon Range':<15}")
    print("   " + "-" * 70)
    
    state_stats = []
    for state in sorted(train_df['state'].unique()):
        state_data = train_df[train_df['state'] == state]
        lat_range = state_data['latitude'].max() - state_data['latitude'].min()
        lon_range = state_data['longitude'].max() - state_data['longitude'].min()
        
        state_stats.append({
            'state': state,
            'count': len(state_data),
            'lat_range': lat_range,
            'lon_range': lon_range,
            'lat_mean': state_data['latitude'].mean(),
            'lon_mean': state_data['longitude'].mean()
        })
        
        if len(state_data) > 100:  # Only show states with decent sample size
            print(f"   {state:<20} {len(state_data):<10} {lat_range:<15.2f} {lon_range:<15.2f}")
    
    return pd.DataFrame(state_stats)


def check_data_quality():
    """Check for data quality issues"""
    
    DATA_DIR = Path("/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset")
    train_df = pd.read_csv(DATA_DIR / "train_ground_truth.csv")
    
    print("\n10. Data quality checks:")
    
    # Check for duplicates
    duplicates = train_df.duplicated(subset=['latitude', 'longitude']).sum()
    print(f"   Duplicate GPS coordinates: {duplicates}")
    
    # Check for outliers
    lat_outliers = ((train_df['latitude'] < 20) | (train_df['latitude'] > 55)).sum()
    lon_outliers = ((train_df['longitude'] < -130) | (train_df['longitude'] > -60)).sum()
    print(f"   Latitude outliers: {lat_outliers}")
    print(f"   Longitude outliers: {lon_outliers}")
    
    # Check state_idx consistency
    state_df = pd.read_csv(DATA_DIR / "state_mapping.csv")
    valid_indices = set(state_df['state_idx'])
    invalid_indices = set(train_df['state_idx']) - valid_indices
    print(f"   Invalid state indices: {len(invalid_indices)}")
    if invalid_indices:
        print(f"   Invalid indices: {invalid_indices}")


if __name__ == "__main__":
    try:
        explore_dataset()
        state_stats = analyze_state_characteristics()
        check_data_quality()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the dataset is in the correct location:")
        print("/Users/pxndey/nyu/f25/csgy6643/project_4/kaggle_dataset/")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()