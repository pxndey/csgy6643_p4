#!/usr/bin/env python3
"""
Profile model forward/backward pass speed to identify bottleneck
"""

import torch
import time
import numpy as np
from transformers import CLIPModel
from pathlib import Path
import pandas as pd

# Test different devices
def test_device_speed(device_name):
    """Test model speed on different devices"""
    
    print(f"\n{'='*60}")
    print(f"Testing on: {device_name}")
    print(f"{'='*60}")
    
    device = torch.device(device_name)
    
    # Load model
    print("Loading model...")
    from train_geoguessr import GeoLocatorModel
    
    clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    model = GeoLocatorModel(
        clip_model=clip_model,
        num_states=33,
        freeze_backbone=True,
        use_4_views=True
    )
    model = model.to(device)
    model.train()
    
    # Create dummy batch
    batch_size = 16
    dummy_input = torch.randn(batch_size, 4, 3, 336, 336).to(device)
    dummy_state = torch.randint(0, 33, (batch_size,)).to(device)
    dummy_gps = torch.randn(batch_size, 2).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        state_logits, gps_pred = model(dummy_input)
        loss = state_logits.sum() + gps_pred.sum()
        loss.backward()
    
    # Time forward pass
    print("Timing forward pass (10 iterations)...")
    forward_times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            state_logits, gps_pred = model(dummy_input)
        if device_name == 'mps':
            torch.mps.synchronize()
        forward_times.append(time.time() - start)
    
    avg_forward = np.mean(forward_times)
    print(f"  Average forward pass: {avg_forward:.3f}s")
    print(f"  Per batch (16 samples): {avg_forward:.3f}s")
    print(f"  Per epoch (3712 batches): {avg_forward * 3712 / 60:.1f} minutes")
    
    # Time backward pass
    print("\nTiming forward + backward pass (10 iterations)...")
    backward_times = []
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    
    for _ in range(10):
        start = time.time()
        optimizer.zero_grad()
        state_logits, gps_pred = model(dummy_input)
        loss = state_logits.sum() + gps_pred.sum()
        loss.backward()
        optimizer.step()
        if device_name == 'mps':
            torch.mps.synchronize()
        backward_times.append(time.time() - start)
    
    avg_backward = np.mean(backward_times)
    print(f"  Average forward+backward: {avg_backward:.3f}s")
    print(f"  Per epoch (3712 batches): {avg_backward * 3712 / 60:.1f} minutes")
    
    # Estimate total training time
    total_minutes_per_epoch = avg_backward * 3712 / 60
    total_hours_15_epochs = total_minutes_per_epoch * 15 / 60
    
    print(f"\n  📊 ESTIMATED TRAINING TIME:")
    print(f"     Per epoch: {total_minutes_per_epoch:.1f} minutes")
    print(f"     15 epochs: {total_hours_15_epochs:.1f} hours")
    
    return {
        'device': device_name,
        'forward_time': avg_forward,
        'backward_time': avg_backward,
        'epoch_minutes': total_minutes_per_epoch
    }


def main():
    print("="*60)
    print("MODEL PERFORMANCE PROFILING")
    print("="*60)
    
    results = []
    
    # Test MPS
    if torch.backends.mps.is_available():
        try:
            results.append(test_device_speed('mps'))
        except Exception as e:
            print(f"MPS failed: {e}")
    
    # Test CPU
    try:
        results.append(test_device_speed('cpu'))
    except Exception as e:
        print(f"CPU failed: {e}")
    
    # Compare results
    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        for r in results:
            print(f"{r['device']:6s}: {r['epoch_minutes']:6.1f} min/epoch " + 
                  f"({r['epoch_minutes']*15/60:.1f}h for 15 epochs)")
        
        if results[0]['epoch_minutes'] > results[1]['epoch_minutes']:
            faster = results[1]
            print(f"\n✅ {faster['device'].upper()} is FASTER - use device='{faster['device']}'")
        else:
            faster = results[0]
            print(f"\n✅ {faster['device'].upper()} is FASTER - use device='{faster['device']}'")


if __name__ == "__main__":
    main()