"""Unit tests for configuration management."""

import pytest
import tempfile
import os
import json
from src.config import ConfigManager, IntersectionConfig
from src.config.config_manager import SystemConfig


class TestSystemConfig:
    """Test cases for SystemConfig."""
    
    def test_default_system_config(self):
        """Test default system configuration values."""
        config = SystemConfig()
        
        assert config.camera_fps == 10
        assert config.detection_confidence_threshold == 0.8
        assert config.learning_rate == 0.01
        assert config.prediction_horizon_minutes == 30
        assert config.log_level == "INFO"


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization with default config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            manager = ConfigManager(config_file)
            
            assert isinstance(manager.get_system_config(), SystemConfig)
            assert len(manager.get_all_intersection_ids()) > 0
    
    def test_update_system_config(self):
        """Test updating system configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            manager = ConfigManager(config_file)
            
            manager.update_system_config(camera_fps=15, learning_rate=0.02)
            
            config = manager.get_system_config()
            assert config.camera_fps == 15
            assert config.learning_rate == 0.02
    
    def test_intersection_config_management(self):
        """Test adding and retrieving intersection configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            manager = ConfigManager(config_file)
            
            test_config = {
                'name': 'Test Intersection',
                'geometry': {
                    'lanes': {'north': {'count': 2, 'length': 100}},
                    'signal_phases': ['north_south_green'],
                    'default_phase_timings': {'north_south_green': 45}
                },
                'camera_positions': {
                    'north': {'x': 0, 'y': 50, 'angle': 180}
                }
            }
            
            manager.add_intersection_config("test_intersection", test_config)
            retrieved_config = manager.get_intersection_config("test_intersection")
            
            assert retrieved_config is not None
            assert retrieved_config['name'] == 'Test Intersection'
    
    def test_config_validation(self):
        """Test intersection configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            manager = ConfigManager(config_file)
            
            # Valid config
            valid_config = {
                'name': 'Test Intersection',
                'geometry': {
                    'lanes': {'north': {'count': 2, 'length': 100}},
                    'signal_phases': ['north_south_green']
                },
                'camera_positions': {
                    'north': {'x': 0, 'y': 50, 'angle': 180}
                }
            }
            
            assert manager.validate_intersection_config(valid_config) == True
            
            # Invalid config (missing required key)
            invalid_config = {
                'name': 'Test Intersection',
                'geometry': {
                    'lanes': {'north': {'count': 2, 'length': 100}}
                    # Missing signal_phases
                }
            }
            
            assert manager.validate_intersection_config(invalid_config) == False
    
    def test_config_persistence(self):
        """Test saving and loading configuration from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            
            # Create and configure manager
            manager1 = ConfigManager(config_file)
            manager1.update_system_config(camera_fps=20)
            manager1.save_config()
            
            # Create new manager and verify config was loaded
            manager2 = ConfigManager(config_file)
            config = manager2.get_system_config()
            
            assert config.camera_fps == 20


class TestIntersectionConfig:
    """Test cases for IntersectionConfig."""
    
    def test_intersection_config_creation(self):
        """Test creating an intersection configuration."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        assert config.intersection_id == "test_001"
        assert config.name == "Test Intersection"
        assert len(config.lanes) == 0
        assert len(config.signal_phases) == 0
    
    def test_add_lane_configuration(self):
        """Test adding lane configurations."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        config.add_lane("north", 2, 100.0)
        config.add_lane("south", 3, 120.0)
        
        assert config.get_lane_count("north") == 2
        assert config.get_lane_length("north") == 100.0
        assert config.get_lane_count("south") == 3
        assert config.get_total_lanes() == 5
    
    def test_invalid_lane_configuration(self):
        """Test validation of invalid lane configurations."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        with pytest.raises(ValueError, match="Lane count must be a positive integer"):
            config.add_lane("north", 0, 100.0)
        
        with pytest.raises(ValueError, match="Lane length must be a positive number"):
            config.add_lane("north", 2, -50.0)
    
    def test_camera_position_configuration(self):
        """Test adding camera position configurations."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        config.add_camera_position("north", 0.0, 50.0, 180.0)
        config.add_camera_position("south", 0.0, -50.0, 0.0)
        
        assert config.has_camera("north") == True
        assert config.has_camera("east") == False
        
        north_pos = config.get_camera_position("north")
        assert north_pos == (0.0, 50.0, 180.0)
    
    def test_invalid_camera_position(self):
        """Test validation of invalid camera positions."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        with pytest.raises(ValueError, match="Camera angle must be between 0 and 360 degrees"):
            config.add_camera_position("north", 0.0, 50.0, 400.0)
    
    def test_signal_phase_configuration(self):
        """Test signal phase configuration."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        phases = ["north_south_green", "east_west_green"]
        config.set_signal_phases(phases)
        
        assert config.signal_phases == phases
        
        timings = {"north_south_green": 45, "east_west_green": 40, "yellow": 3}
        config.set_default_phase_timings(timings)
        
        assert config.default_phase_timings == timings
    
    def test_invalid_signal_phases(self):
        """Test validation of invalid signal phases."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        with pytest.raises(ValueError, match="Signal phases must be a non-empty list"):
            config.set_signal_phases([])
        
        with pytest.raises(ValueError, match="Each signal phase must be a non-empty string"):
            config.set_signal_phases(["valid_phase", ""])
    
    def test_invalid_phase_timings(self):
        """Test validation of invalid phase timings."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        with pytest.raises(ValueError, match="Phase timings must be a non-empty dictionary"):
            config.set_default_phase_timings({})
        
        with pytest.raises(ValueError, match="Timing for phase test_phase must be a positive integer"):
            config.set_default_phase_timings({"test_phase": 0})
    
    def test_intersection_validation(self):
        """Test complete intersection configuration validation."""
        config = IntersectionConfig("test_001", "Test Intersection")
        
        # Should fail validation - missing required components
        with pytest.raises(ValueError, match="At least one lane direction must be configured"):
            config.validate()
        
        # Add required components
        config.add_lane("north", 2, 100.0)
        config.set_signal_phases(["north_south_green"])
        config.set_default_phase_timings({"north_south_green": 45})
        
        # Should pass validation
        config.validate()
    
    def test_to_dict_conversion(self):
        """Test converting intersection config to dictionary."""
        config = IntersectionConfig("test_001", "Test Intersection")
        config.add_lane("north", 2, 100.0)
        config.add_camera_position("north", 0.0, 50.0, 180.0)
        config.set_signal_phases(["north_south_green"])
        config.set_default_phase_timings({"north_south_green": 45})
        
        config_dict = config.to_dict()
        
        assert config_dict['name'] == "Test Intersection"
        assert config_dict['geometry']['lanes']['north']['count'] == 2
        assert config_dict['camera_positions']['north']['x'] == 0.0