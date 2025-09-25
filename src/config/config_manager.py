"""Configuration manager for system parameters."""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """System-wide configuration parameters."""
    
    # Camera processing settings
    camera_fps: int = 10
    detection_confidence_threshold: float = 0.8
    vehicle_detection_enabled: bool = True
    
    # RL Agent settings
    learning_rate: float = 0.01
    epsilon: float = 0.1
    discount_factor: float = 0.95
    
    # Prediction settings
    prediction_horizon_minutes: int = 30
    prediction_update_interval_seconds: int = 60
    min_prediction_confidence: float = 0.7
    
    # Signal control settings
    max_signal_adjustment_seconds: int = 120
    signal_update_interval_seconds: int = 30
    emergency_override_timeout_seconds: int = 300
    
    # Dashboard settings
    dashboard_refresh_rate_seconds: int = 5
    max_historical_data_points: int = 1000
    
    # Logging settings
    log_level: str = "INFO"
    log_file_path: str = "logs/traffic_system.log"


class ConfigManager:
    """Manages system configuration and intersection geometry."""
    
    def __init__(self, config_file: str = "config/system_config.json"):
        self.config_file = config_file
        self.system_config = SystemConfig()
        self.intersection_configs: Dict[str, Dict[str, Any]] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load system config
                if 'system' in config_data:
                    system_data = config_data['system']
                    for key, value in system_data.items():
                        if hasattr(self.system_config, key):
                            setattr(self.system_config, key, value)
                
                # Load intersection configs
                if 'intersections' in config_data:
                    self.intersection_configs = config_data['intersections']
                
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                logger.info("No config file found, using default configuration")
                self._create_default_config()
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            'system': asdict(self.system_config),
            'intersections': {
                'intersection_001': {
                    'name': 'Main St & Oak Ave',
                    'geometry': {
                        'lanes': {
                            'north': {'count': 2, 'length': 100},
                            'south': {'count': 2, 'length': 100},
                            'east': {'count': 2, 'length': 100},
                            'west': {'count': 2, 'length': 100}
                        },
                        'signal_phases': ['north_south_green', 'east_west_green'],
                        'default_phase_timings': {
                            'north_south_green': 45,
                            'east_west_green': 45,
                            'yellow': 3,
                            'all_red': 2
                        }
                    },
                    'camera_positions': {
                        'north': {'x': 0, 'y': 50, 'angle': 180},
                        'south': {'x': 0, 'y': -50, 'angle': 0},
                        'east': {'x': 50, 'y': 0, 'angle': 270},
                        'west': {'x': -50, 'y': 0, 'angle': 90}
                    }
                }
            }
        }
        
        self.intersection_configs = default_config['intersections']
        self.save_config()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            config_data = {
                'system': asdict(self.system_config),
                'intersections': self.intersection_configs
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.system_config
    
    def update_system_config(self, **kwargs) -> None:
        """Update system configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.system_config, key):
                setattr(self.system_config, key, value)
                logger.info(f"Updated system config: {key} = {value}")
            else:
                logger.warning(f"Unknown system config parameter: {key}")
    
    def get_intersection_config(self, intersection_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific intersection."""
        return self.intersection_configs.get(intersection_id)
    
    def add_intersection_config(self, intersection_id: str, config: Dict[str, Any]) -> None:
        """Add or update intersection configuration."""
        self.intersection_configs[intersection_id] = config
        logger.info(f"Added/updated intersection config: {intersection_id}")
    
    def get_all_intersection_ids(self) -> list:
        """Get list of all configured intersection IDs."""
        return list(self.intersection_configs.keys())
    
    def validate_intersection_config(self, config: Dict[str, Any]) -> bool:
        """Validate intersection configuration structure."""
        required_keys = ['name', 'geometry', 'camera_positions']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required key in intersection config: {key}")
                return False
        
        # Validate geometry
        geometry = config['geometry']
        if 'lanes' not in geometry or 'signal_phases' not in geometry:
            logger.error("Invalid geometry configuration")
            return False
        
        return True