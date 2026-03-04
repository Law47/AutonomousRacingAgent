"""
Handles racing line data from AC Lua plugin (CSP_Reset) and calculates distances.

The Lua script writes racing line data to a cache file (~ac_racing_line_cache.bin)
which this manager reads to calculate the car's distance from the ideal racing line.
"""

import numpy as np
import os
import struct
from logging import Logger
from omegaconf import OmegaConf
from typing import List, Tuple


class RacingLineManager:
    """
    Manages racing line data from Assetto Corsa Lua plugin (CSP_Reset).
    
    The Lua script writes racing line coordinates to a cache file which this manager 
    reads to calculate the car's distance from the ideal racing line.
    """
    
    def __init__(self, config: OmegaConf, logger: Logger):
        """
        Initialize racing line manager for Lua plugin-based data.
        
        Config parameters (in racing_line section):
            enable: whether to use racing line data (default: True)
            track_name: track name for logging (informational)
            line_distance_weight: weight of racing line distance penalty
            line_distance_threshold: maximum distance before full penalty (meters)
        
        Reads racing line from: %USERPROFILE%/ac_racing_line_cache.bin
        """
        self.config = config
        self.logger = logger
        
        # Get racing line config
        rl_cfg = config.get('racing_line', {})
        
        self.enabled = rl_cfg.get('enable', True)
        self.track_name = rl_cfg.get('track_name', 'unknown')
        
        # Reward configuration
        self.line_distance_weight = rl_cfg.get('line_distance_weight', 5.0)
        self.line_distance_threshold = rl_cfg.get('line_distance_threshold', 10.0)
        
        # Racing line data (will be populated from plugin)
        self.racing_line_points: List[Tuple[float, float, float]] = []
        self.racing_line_loaded = False
        self._racing_line_cache = None  # Cached numpy array for fast lookup
        
        self.logger.info(f"Racing Line Manager initialized (Lua plugin-based):")
        self.logger.info(f"  Enabled: {self.enabled}")
        self.logger.info(f"  Track: {self.track_name}")
        self.logger.info(f"  Distance weight: {self.line_distance_weight}")
        self.logger.info(f"  Distance threshold: {self.line_distance_threshold}m")
        self.logger.info(f"  Cache file: {os.path.expanduser('~')}/ac_racing_line_cache.bin")
        self.logger.info(f"  Waiting for Lua plugin to write racing line data...")
    
    def update_racing_line_from_lua_cache(self) -> bool:
        """
        Update racing line data from Lua plugin cache file.
        
        The Lua script (CSP_Reset) writes racing line to:
        %USERPROFILE%/ac_racing_line_cache.bin
        
        File format: [point_count (int32)] [point_0 (3x float32)] [point_1 (3x float32)] ...
        
        Returns:
            bool: True if racing line was successfully loaded
        """
        if not self.enabled:
            return False
        
        try:
            home = os.path.expanduser("~")
            cache_file = os.path.join(home, "ac_racing_line_cache.bin")
            
            if not os.path.exists(cache_file):
                if not self.racing_line_loaded:
                    if not hasattr(self, '_logged_cache_missing'):
                        self.logger.warning(f"[RACING_LINE] Cache file not found: {cache_file}")
                        self.logger.warning(f"[RACING_LINE] Verify CSP_Reset script is ENABLED in Content Manager")
                        self._logged_cache_missing = True
                return False
            
            # Read cache file
            with open(cache_file, "rb") as f:
                # Read point count (4 bytes, little-endian int32)
                point_count_bytes = f.read(4)
                if len(point_count_bytes) < 4:
                    if not self.racing_line_loaded:
                        self.logger.warning(f"[RACING_LINE] Cache file corrupted (can't read point count)")
                    return False
                
                point_count = struct.unpack("<i4", point_count_bytes)[0]
                
                if point_count <= 0 or point_count > 500:
                    if not self.racing_line_loaded:
                        self.logger.warning(f"[RACING_LINE] Invalid point count in cache: {point_count}")
                    return False
                
                # Read all points (each point is 3 floats: x, y, z)
                self.racing_line_points = []
                bytes_read = 0
                for i in range(point_count):
                    point_data = f.read(12)  # 3 floats * 4 bytes each
                    if len(point_data) < 12:
                        self.logger.warning(f"[RACING_LINE] Incomplete point data at index {i} (read {len(point_data)}/12 bytes)")
                        break
                    x, y, z = struct.unpack("<fff", point_data)
                    self.racing_line_points.append((float(x), float(y), float(z)))
                    bytes_read += 12
                
                if len(self.racing_line_points) > 0:
                    # Convert to numpy array for fast calculations
                    self._racing_line_cache = np.array(self.racing_line_points)
                    
                    if not self.racing_line_loaded:
                        self.racing_line_loaded = True
                        self.logger.info(f"✓ Racing line loaded: {len(self.racing_line_points)} points ({bytes_read} bytes)")
                        if len(self.racing_line_points) > 0:
                            self.logger.info(f"  First point: {self.racing_line_points[0]}")
                            self.logger.info(f"  Last point: {self.racing_line_points[-1]}")
                    
                    return True
                else:
                    if not self.racing_line_loaded:
                        self.logger.warning(f"[RACING_LINE] No points parsed from cache file")
                    return False
        
        except Exception as e:
            if not self.racing_line_loaded:
                self.logger.warning(f"[RACING_LINE] Error reading cache: {type(e).__name__}: {e}")
            return False
    
    def distance_to_racing_line(self, car_pos: Tuple[float, float, float]) -> float:
        """
        Calculate distance from car to racing line.
        
        Args:
            car_pos: (x, y, z) position of car
        
        Returns:
            float: Distance to racing line in meters.
                   Returns threshold value if data unavailable (default penalty).
        """
        if not self.enabled or not self.racing_line_loaded:
            return self.line_distance_threshold  # Default penalty
        
        if self._racing_line_cache is None or len(self._racing_line_cache) == 0:
            return self.line_distance_threshold
        
        try:
            car_pos_array = np.array(car_pos)
            
            # Find closest point on racing line using vectorized calculation
            distances = np.linalg.norm(self._racing_line_cache - car_pos_array, axis=1)
            min_distance = float(np.min(distances))
            
            return min_distance
        
        except Exception as e:
            self.logger.debug(f"Error calculating racing line distance: {e}")
            return self.line_distance_threshold
    
    def get_line_visualization_points(self, max_points: int = 100) -> List[Tuple[float, float, float]]:
        """
        Get racing line points for visualization (e.g., debug overlay).
        
        Args:
            max_points: Maximum number of points to return (for performance)
        
        Returns:
            List of (x, y, z) coordinates representing racing line
        """
        if not self.racing_line_loaded or not self.racing_line_points:
            return []
        
        # Sample points evenly along the line
        if len(self.racing_line_points) <= max_points:
            return self.racing_line_points
        
        indices = np.linspace(0, len(self.racing_line_points) - 1, max_points, dtype=int)
        return [self.racing_line_points[i] for i in indices]
