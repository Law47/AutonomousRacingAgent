"""
Handles racing line data from CSV files and calculates distances.

Racing line CSV files are generated from Assetto Corsa .ai files and contain
the ideal path around the track at each point.

CSV Format:
    pos_x, pos_y, left_border_x, left_border_y, right_border_x, right_border_y,
    throttle, brake, target_speed, yaw
"""

import numpy as np
import pandas as pd
import os
from logging import Logger
from omegaconf import OmegaConf
from typing import Tuple


class RacingLineManager:
    """
    Manages racing line data from pre-generated CSV files.
    
    Loads racing line coordinates from CSV and provides efficient distance
    calculations from any car position to the nearest point on the racing line.
    """
    
    def __init__(self, config: OmegaConf, logger: Logger):
        """
        Initialize racing line manager for CSV-based data.
        
        Config parameters (in racing_line section):
            enable: whether to use racing line data (default: True)
            track_name: track name to identify CSV file
            line_distance_weight: weight of racing line distance penalty
            line_distance_threshold: distance at which full penalty applies (meters)
        
        Loads racing line from: Racing Lines/{track_name}_racing_line.csv
        """
        self.config = config
        self.logger = logger
        
        # Get racing line config
        rl_cfg = config.get('racing_line', {})
        
        self.enabled = rl_cfg.get('enable', True)
        self.track_name = rl_cfg.get('track_name', 'monza')
        
        # Reward configuration
        self.line_distance_weight = rl_cfg.get('line_distance_weight', 5.0)
        self.line_distance_threshold = rl_cfg.get('line_distance_threshold', 10.0)
        
        # Racing line data (will be populated from CSV)
        self.racing_line_points = None  # numpy array of shape (N, 2) with (x, y)
        self.racing_line_loaded = False
        
        self.logger.info(f"Racing Line Manager initialized (CSV-based):")
        self.logger.info(f"  Enabled: {self.enabled}")
        self.logger.info(f"  Track: {self.track_name}")
        self.logger.info(f"  Distance weight: {self.line_distance_weight}")
        self.logger.info(f"  Distance threshold: {self.line_distance_threshold}m")
        
        # Try to load racing line on initialization
        if self.enabled:
            self._load_racing_line_from_csv()
    
    def _load_racing_line_from_csv(self) -> bool:
        """
        Load racing line data from CSV file.
        
        Looks for: Racing Lines/{track_name}_racing_line.csv
        
        Returns:
            bool: True if racing line was successfully loaded
        """
        try:
            # Find Racing Lines directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            racing_lines_dir = os.path.join(current_dir, "Racing Lines")
            csv_file = os.path.join(racing_lines_dir, f"{self.track_name}_racing_line.csv")
            
            if not os.path.exists(csv_file):
                self.logger.warning(f"[RACING_LINE] CSV file not found: {csv_file}")
                self.logger.warning(f"[RACING_LINE] Available racing lines in: {racing_lines_dir}")
                if os.path.exists(racing_lines_dir):
                    files = os.listdir(racing_lines_dir)
                    self.logger.warning(f"[RACING_LINE] Files: {files}")
                return False
            
            # Load CSV
            df = pd.read_csv(csv_file)
            
            # Extract racing line points (x, y coordinates)
            self.racing_line_points = df[['pos_x', 'pos_y']].values.astype(np.float32)
            
            if len(self.racing_line_points) == 0:
                self.logger.warning(f"[RACING_LINE] CSV file is empty: {csv_file}")
                return False
            
            self.racing_line_loaded = True
            self.logger.info(f"✓ Racing line loaded: {len(self.racing_line_points)} points from {csv_file}")
            self.logger.info(f"  Track bounds: X=[{self.racing_line_points[:, 0].min():.1f}, {self.racing_line_points[:, 0].max():.1f}] "
                           f"Y=[{self.racing_line_points[:, 1].min():.1f}, {self.racing_line_points[:, 1].max():.1f}]")
            
            return True
        
        except Exception as e:
            self.logger.error(f"[RACING_LINE] Error loading CSV: {type(e).__name__}: {e}")
            return False
    
    def distance_to_racing_line(self, car_pos: Tuple[float, float, float]) -> float:
        """
        Calculate distance from car to nearest point on racing line.
        
        Uses vectorized numpy operations for efficiency.
        
        Args:
            car_pos: (x, y, z) position of car (z is ignored)
        
        Returns:
            float: Distance to racing line in meters.
                   Returns threshold value if data unavailable (default penalty).
        """
        if not self.enabled or not self.racing_line_loaded:
            return self.line_distance_threshold
        
        if self.racing_line_points is None or len(self.racing_line_points) == 0:
            return self.line_distance_threshold
        
        try:
            # Extract x, y from car position (ignore z)
            car_xy = np.array([car_pos[0], car_pos[1]], dtype=np.float32)
            
            # Vectorized distance calculation to all points on racing line
            distances = np.linalg.norm(self.racing_line_points - car_xy, axis=1)
            min_distance = float(np.min(distances))
            
            return min_distance
        
        except Exception as e:
            self.logger.debug(f"Error calculating racing line distance: {e}")
            return self.line_distance_threshold
    
    def get_line_visualization_points(self, max_points: int = 100):
        """
        Get racing line points for visualization or debugging.
        
        Args:
            max_points: Maximum number of points to return (for performance)
        
        Returns:
            List of (x, y) coordinates or empty list if not loaded
        """
        if not self.racing_line_loaded or self.racing_line_points is None:
            return []
        
        # Sample points evenly along the line
        if len(self.racing_line_points) <= max_points:
            return self.racing_line_points.tolist()
        
        indices = np.linspace(0, len(self.racing_line_points) - 1, max_points, dtype=int)
        return self.racing_line_points[indices].tolist()
