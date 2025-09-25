"""Tests for traffic simulator."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.processors.traffic_simulator import TrafficSimulator, TrafficPattern
from src.models.vehicle_detection import VehicleDetection


class TestTrafficPattern:
    """Test TrafficPattern dataclass."""
    
    def test_traffic_pattern_creation(self):
        """Test creating a traffic pattern."""
        pattern = TrafficPattern(base_flow_rate=10.0)
        
        assert pattern.base_flow_rate == 10.0
        assert pattern.peak_multiplier == 1.5
        assert pattern.vehicle_type_distribution is not None
        assert 'car' in pattern.vehicle_type_distribution
        assert pattern.speed_range == (20.0, 60.0)
    
    def test_traffic_pattern_custom_distribution(self):
        """Test traffic pattern with custom vehicle distribution."""
        custom_dist = {'car': 0.9, 'truck': 0.1}
        pattern = TrafficPattern(
            base_flow_rate=15.0,
            vehicle_type_distribution=custom_dist
        )
        
        assert pattern.vehicle_type_distribution == custom_dist


class TestTrafficSimulator:
    """Test TrafficSimulator class."""
    
    @pytest.fixture
    def intersection_config(self):
        """Sample intersection configuration."""
        return {
            'geometry': {
                'lanes': {
                    'north': {'count': 2, 'length': 100},
                    'south': {'count': 2, 'length': 100},
                    'east': {'count': 2, 'length': 100},
                    'west': {'count': 2, 'length': 100}
                }
            }
        }
    
    @pytest.fixture
    def simulator(self, intersection_config):
        """Create traffic simulator instance."""
        return TrafficSimulator(intersection_config)
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.intersection_config is not None
        assert simulator.vehicle_counter == 0
        assert len(simulator.patterns) == 4
        assert 'morning_rush' in simulator.patterns
        assert 'evening_rush' in simulator.patterns
        assert 'midday' in simulator.patterns
        assert 'night' in simulator.patterns
    
    def test_get_current_pattern_morning_rush(self, simulator):
        """Test getting morning rush pattern."""
        morning_time = datetime(2023, 1, 1, 8, 0)  # 8 AM
        pattern = simulator.get_current_pattern(morning_time)
        
        assert pattern == simulator.patterns['morning_rush']
    
    def test_get_current_pattern_evening_rush(self, simulator):
        """Test getting evening rush pattern."""
        evening_time = datetime(2023, 1, 1, 18, 0)  # 6 PM
        pattern = simulator.get_current_pattern(evening_time)
        
        assert pattern == simulator.patterns['evening_rush']
    
    def test_get_current_pattern_midday(self, simulator):
        """Test getting midday pattern."""
        midday_time = datetime(2023, 1, 1, 14, 0)  # 2 PM
        pattern = simulator.get_current_pattern(midday_time)
        
        assert pattern == simulator.patterns['midday']
    
    def test_get_current_pattern_night(self, simulator):
        """Test getting night pattern."""
        night_time = datetime(2023, 1, 1, 2, 0)  # 2 AM
        pattern = simulator.get_current_pattern(night_time)
        
        assert pattern == simulator.patterns['night']
    
    def test_generate_vehicle_detections(self, simulator):
        """Test generating vehicle detections."""
        timestamp = datetime.now()
        detections = simulator.generate_vehicle_detections(timestamp, duration_seconds=5)
        
        assert isinstance(detections, list)
        assert len(detections) > 0
        
        # Check detection properties
        for detection in detections:
            assert isinstance(detection, VehicleDetection)
            assert detection.vehicle_type in ['car', 'truck', 'bus', 'motorcycle']
            assert detection.confidence >= 0.85
            assert detection.timestamp >= timestamp
    
    def test_generate_vehicle_detections_empty_config(self):
        """Test generating detections with empty config."""
        empty_config = {'geometry': {'lanes': {}}}
        simulator = TrafficSimulator(empty_config)
        
        timestamp = datetime.now()
        detections = simulator.generate_vehicle_detections(timestamp)
        
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_simulate_accident_event(self, simulator):
        """Test simulating accident event."""
        timestamp = datetime.now()
        detections = simulator.simulate_traffic_event('accident', timestamp)
        
        assert isinstance(detections, list)
        assert len(detections) >= 5  # Should generate at least 5 vehicles
        
        # Check that vehicles are clustered (similar positions)
        if len(detections) > 1:
            positions = [d.position for d in detections]
            # Vehicles should be in similar area (accident backup)
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Check that there's some clustering (not too spread out)
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            assert x_range < 50 or y_range < 50  # Clustered in at least one dimension
    
    def test_simulate_rush_hour_spike(self, simulator):
        """Test simulating rush hour spike."""
        timestamp = datetime.now()
        detections = simulator.simulate_traffic_event('rush_hour_spike', timestamp)
        
        assert isinstance(detections, list)
        assert len(detections) > 20  # Should generate many vehicles
        
        # Check that vehicles are distributed across lanes
        lanes = set(d.lane for d in detections)
        assert len(lanes) > 1  # Multiple lanes should have vehicles
    
    def test_get_traffic_statistics_empty(self, simulator):
        """Test getting statistics with no vehicles."""
        stats = simulator.get_traffic_statistics()
        
        assert stats['total_vehicles'] == 0
        assert stats['vehicles_by_type'] == {}
        assert stats['vehicles_by_lane'] == {}
        assert stats['average_confidence'] == 0.0
    
    def test_get_traffic_statistics_with_vehicles(self, simulator):
        """Test getting statistics with vehicles."""
        # Generate some detections first
        timestamp = datetime.now()
        detections = simulator.generate_vehicle_detections(timestamp)
        
        stats = simulator.get_traffic_statistics()
        
        assert stats['total_vehicles'] > 0
        assert isinstance(stats['vehicles_by_type'], dict)
        assert isinstance(stats['vehicles_by_lane'], dict)
        assert 0.0 <= stats['average_confidence'] <= 1.0
    
    def test_clear_old_vehicles(self, simulator):
        """Test clearing old vehicles."""
        # Generate some detections
        timestamp = datetime.now()
        simulator.generate_vehicle_detections(timestamp)
        
        initial_count = len(simulator.active_vehicles)
        assert initial_count > 0
        
        # Clear vehicles older than future time (should clear all)
        # Add extra time to account for random timestamp additions in generation
        cutoff_time = timestamp + timedelta(seconds=10)
        simulator.clear_old_vehicles(cutoff_time)
        
        assert len(simulator.active_vehicles) == 0
    
    def test_vehicle_position_generation(self, simulator):
        """Test vehicle position generation for different lanes."""
        lane_config = {'length': 100}
        
        # Test different lane directions
        for lane in ['north', 'south', 'east', 'west']:
            position = simulator._generate_vehicle_position(lane, lane_config)
            
            assert isinstance(position, tuple)
            assert len(position) == 2
            assert isinstance(position[0], (int, float))
            assert isinstance(position[1], (int, float))
            
            # Check that positions are reasonable for each direction
            x, y = position
            if lane == 'north':
                assert x == 0
                assert y >= 0
            elif lane == 'south':
                assert x == 0
                assert y <= 0
            elif lane == 'east':
                assert x >= 0
                assert y == 0
            elif lane == 'west':
                assert x <= 0
                assert y == 0
    
    @patch('random.random')
    def test_generate_vehicle_type_distribution(self, mock_random, simulator):
        """Test vehicle type generation follows distribution."""
        pattern = simulator.patterns['midday']
        
        # Test car generation (first in distribution)
        mock_random.return_value = 0.1  # Should select 'car'
        vehicle_type = simulator._generate_vehicle_type(pattern)
        assert vehicle_type == 'car'
        
        # Test truck generation
        mock_random.return_value = 0.8  # Should select 'truck'
        vehicle_type = simulator._generate_vehicle_type(pattern)
        assert vehicle_type == 'truck'