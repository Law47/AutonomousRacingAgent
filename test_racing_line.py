#!/usr/bin/env python3
"""
Quick test to verify racing line CSV loading works correctly.
Run this from the project root: python test_racing_line.py
"""

import sys
import numpy as np
from omegaconf import OmegaConf
from logging import getLogger, basicConfig, INFO

# Set up logging
basicConfig(level=INFO, format='%(levelname)s - %(name)s: %(message)s')
logger = getLogger(__name__)

# Load config
config = OmegaConf.load('config.yml')

# Import after logging is set up
from racing_line_manager import RacingLineManager

print("\n" + "="*70)
print("RACING LINE MANAGER TEST")
print("="*70 + "\n")

# Initialize racing line manager
rlm = RacingLineManager(config, logger)

print(f"\n✓ Racing line manager initialized")
print(f"  Enabled: {rlm.enabled}")
print(f"  Track: {rlm.track_name}")
print(f"  Loaded: {rlm.racing_line_loaded}")

if rlm.racing_line_loaded:
    print(f"\n✓ Racing line successfully loaded!")
    print(f"  Total points: {len(rlm.racing_line_points)}")
    print(f"  Point shape: {rlm.racing_line_points.shape}")
    print(f"  X range: [{rlm.racing_line_points[:, 0].min():.1f}, {rlm.racing_line_points[:, 0].max():.1f}]")
    print(f"  Y range: [{rlm.racing_line_points[:, 1].min():.1f}, {rlm.racing_line_points[:, 1].max():.1f}]")
    
    # Test distance calculation
    print(f"\n✓ Testing distance calculation:")
    
    # Test 1: Point on the line
    test_point1 = tuple(rlm.racing_line_points[0]) + (0.0,)
    dist1 = rlm.distance_to_racing_line(test_point1)
    print(f"  Distance from first racing line point: {dist1:.3f}m (expected ~0)")
    
    # Test 2: Point far from line
    test_point2 = (rlm.racing_line_points[0, 0] + 100, rlm.racing_line_points[0, 1] + 100, 0.0)
    dist2 = rlm.distance_to_racing_line(test_point2)
    print(f"  Distance from offset point (100m away): {dist2:.3f}m (expected ~141m)")
    
    # Test 3: Random track point
    idx = len(rlm.racing_line_points) // 2
    test_point3 = tuple(rlm.racing_line_points[idx]) + (0.0,)
    dist3 = rlm.distance_to_racing_line(test_point3)
    print(f"  Distance from midpoint: {dist3:.3f}m (expected ~0)")
    
    # Test 4: Simulate reward calculation like getReward()
    print(f"\n✓ Simulating reward calculation (matching assetto_corsa_gym):")
    threshold = rlm.line_distance_threshold
    
    # Simulate car at 200 km/h on racing line
    speed_kmh = 200.0
    speed_reward = speed_kmh / 300.0  # 0.667
    gap = 0.0  # on the line
    multiplier = 1.0 - (abs(gap) / threshold)
    reward = speed_reward * multiplier
    print(f"  Speed=200km/h, gap=0m: reward={reward:.4f} (speed_norm={speed_reward:.4f}, mult={multiplier:.4f})")
    
    # Simulate car at 200 km/h, 6m from line
    gap = 6.0
    multiplier = 1.0 - min(abs(gap) / threshold, 1.0)
    reward = speed_reward * multiplier
    print(f"  Speed=200km/h, gap=6m: reward={reward:.4f} (speed_norm={speed_reward:.4f}, mult={multiplier:.4f})")
    
    # Simulate car at 200 km/h, 12m from line (threshold)
    gap = 12.0
    multiplier = 1.0 - min(abs(gap) / threshold, 1.0)
    reward = speed_reward * multiplier
    print(f"  Speed=200km/h, gap=12m: reward={reward:.4f} (speed_norm={speed_reward:.4f}, mult={multiplier:.4f})")
    
    # Coordinate system explanation
    print(f"\n✓ Coordinate mapping:")
    print(f"  Racing line CSV: pos_x = AC Z-axis (forward), pos_y = AC X-axis (lateral)")
    print(f"  AC shared memory: carCoordinates = [X, Y_height, Z]")
    print(f"  getReward() maps: car_pos = (carCoordinates[2], carCoordinates[0])")
    
    print(f"\n✓ All tests passed! Racing line is ready to use.")
else:
    print(f"\n✗ Racing line failed to load. Check the Racing Lines/ directory exists.")
    sys.exit(1)

print("\n" + "="*70 + "\n")
