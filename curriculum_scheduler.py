"""
Curriculum Learning Scheduler

Manages curriculum learning progression during training.
- Track training progress (steps/episodes)
- Scale rewards based on curriculum phase
- Support multiple curriculum strategies (linear decay, step-wise, exponential)
"""

import numpy as np
from omegaconf import OmegaConf
from logging import Logger


class CurriculumScheduler:
    """
    Manages curriculum learning progression.
    
    Usage example:
        scheduler = CurriculumScheduler(config, logger)
        # During training loop:
        scheduler.step()  # Call after each environment step
        
        # Get current curriculum weight (0.0 to 1.0)
        racing_line_weight = scheduler.get_racing_line_weight()
        
        # At reset
        scheduler.reset_episode()
    """
    
    def __init__(self, config: OmegaConf, logger: Logger):
        """
        Initialize curriculum scheduler.
        
        Config parameters (in curriculum section):
            strategy: 'linear', 'exponential', or 'step'
            initial_weight: starting weight for racing line reward (0-1)
            final_weight: ending weight (0-1) 
            decay_steps: number of training steps to decay over
            enable: whether curriculum learning is enabled
        """
        self.config = config
        self.logger = logger
        
        # Get curriculum config, with safe defaults
        curriculum_cfg = config.get('curriculum', {})
        
        self.enabled = curriculum_cfg.get('enable', True)
        self.strategy = curriculum_cfg.get('strategy', 'linear')
        self.initial_weight = curriculum_cfg.get('initial_weight', 1.0)
        self.final_weight = curriculum_cfg.get('final_weight', 0.1)
        self.decay_steps = curriculum_cfg.get('decay_steps', 100_000)
        
        # Validate
        self.initial_weight = np.clip(self.initial_weight, 0.0, 1.0)
        self.final_weight = np.clip(self.final_weight, 0.0, 1.0)
        
        self.current_step = 0
        self.current_episode = 0
        self._last_logged_step = 0
        
        self.logger.info(f"Curriculum Scheduler initialized:")
        self.logger.info(f"  Enabled: {self.enabled}")
        self.logger.info(f"  Strategy: {self.strategy}")
        self.logger.info(f"  Weight range: {self.initial_weight} -> {self.final_weight}")
        self.logger.info(f"  Decay over: {self.decay_steps} steps")
    
    def step(self):
        """Call after each environment step during training"""
        self.current_step += 1
        
        # Log progress every 10k steps
        if self.current_step - self._last_logged_step >= 10_000:
            weight = self.get_racing_line_weight()
            self.logger.info(f"Curriculum progress: step {self.current_step}, "
                           f"racing_line_weight={weight:.3f}")
            self._last_logged_step = self.current_step
    
    def reset_episode(self):
        """Call when environment resets (start of new episode)"""
        self.current_episode += 1
    
    def get_racing_line_weight(self) -> float:
        """
        Get current racing line reward weight (0.0 to 1.0).
        
        Returns:
            float: Weight to apply to racing line distance reward.
                   1.0 = full racing line bonus
                   0.0 = no racing line bonus
        """
        if not self.enabled:
            return 0.0
        
        # Calculate progress (0.0 to 1.0+)
        progress = self.current_step / self.decay_steps
        progress = np.clip(progress, 0.0, 1.0)
        
        # Apply decay strategy
        if self.strategy == 'linear':
            # Linear decay from initial to final weight
            weight = self.initial_weight - (self.initial_weight - self.final_weight) * progress
        
        elif self.strategy == 'exponential':
            # Exponential decay (faster initially, slower later)
            weight = self.final_weight + (self.initial_weight - self.final_weight) * np.exp(-3.0 * progress)
        
        elif self.strategy == 'step':
            # Step-wise decay at 25%, 50%, 75% of decay_steps
            if progress < 0.25:
                weight = self.initial_weight
            elif progress < 0.5:
                weight = self.initial_weight * 0.75 + self.final_weight * 0.25
            elif progress < 0.75:
                weight = self.initial_weight * 0.5 + self.final_weight * 0.5
            else:
                weight = self.final_weight
        
        else:
            self.logger.warning(f"Unknown curriculum strategy: {self.strategy}, using linear")
            weight = self.initial_weight - (self.initial_weight - self.final_weight) * progress
        
        return float(np.clip(weight, 0.0, 1.0))
    
    def get_other_curriculum_weights(self) -> dict:
        """
        Get weights for other curriculum components (future use).
        
        Returns:
            dict: Dictionary of component names to weights
        """
        return {
            'racing_line': self.get_racing_line_weight(),
            # Add other curriculum components here as needed
        }
