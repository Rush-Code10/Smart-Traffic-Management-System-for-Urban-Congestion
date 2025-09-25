"""Unit tests for data models."""

import pytest
from datetime import datetime
from src.models import TrafficState, VehicleDetection, SignalAction


class TestTrafficState:
    """Test cases for TrafficState model."""
    
    def test_valid_traffic_state(self):
        """Test creating a valid traffic state."""
        traffic_state = TrafficState(
            intersection_id="test_001",
            timestamp=datetime.now(),
            vehicle_counts={"north": 5, "south": 3, "east": 2, "west": 4},
            queue_lengths={"north": 25.5, "south": 15.0, "east": 10.2, "west": 20.8},
            wait_times={"north": 45.2, "south": 30.1, "east": 25.5, "west": 35.7},
            signal_phase="north_south_green",
            prediction_confidence=0.85
        )
        
        assert traffic_state.intersection_id == "test_001"
        assert traffic_state.signal_phase == "north_south_green"
        assert traffic_state.prediction_confidence == 0.85
    
    def test_invalid_intersection_id(self):
        """Test validation with invalid intersection ID."""
        with pytest.raises(ValueError, match="intersection_id must be a non-empty string"):
            TrafficState(
                intersection_id="",
                timestamp=datetime.now(),
                vehicle_counts={"north": 5},
                queue_lengths={"north": 25.5},
                wait_times={"north": 45.2},
                signal_phase="green"
            )
    
    def test_invalid_vehicle_counts(self):
        """Test validation with invalid vehicle counts."""
        with pytest.raises(ValueError, match="vehicle count for north must be a non-negative integer"):
            TrafficState(
                intersection_id="test_001",
                timestamp=datetime.now(),
                vehicle_counts={"north": -1},
                queue_lengths={"north": 25.5},
                wait_times={"north": 45.2},
                signal_phase="green"
            )
    
    def test_invalid_prediction_confidence(self):
        """Test validation with invalid prediction confidence."""
        with pytest.raises(ValueError, match="prediction_confidence must be a number between 0.0 and 1.0"):
            TrafficState(
                intersection_id="test_001",
                timestamp=datetime.now(),
                vehicle_counts={"north": 5},
                queue_lengths={"north": 25.5},
                wait_times={"north": 45.2},
                signal_phase="green",
                prediction_confidence=1.5
            )
    
    def test_helper_methods(self):
        """Test helper methods of TrafficState."""
        traffic_state = TrafficState(
            intersection_id="test_001",
            timestamp=datetime.now(),
            vehicle_counts={"north": 5, "south": 3},
            queue_lengths={"north": 25.5, "south": 15.0},
            wait_times={"north": 40.0, "south": 30.0},
            signal_phase="green"
        )
        
        assert traffic_state.get_total_vehicles() == 8
        assert traffic_state.get_total_queue_length() == 40.5
        assert traffic_state.get_average_wait_time() == 35.0


class TestVehicleDetection:
    """Test cases for VehicleDetection model."""
    
    def test_valid_vehicle_detection(self):
        """Test creating a valid vehicle detection."""
        detection = VehicleDetection(
            vehicle_id="vehicle_123",
            vehicle_type="car",
            position=(100.5, 200.3),
            lane="north_lane_1",
            confidence=0.92,
            timestamp=datetime.now()
        )
        
        assert detection.vehicle_id == "vehicle_123"
        assert detection.vehicle_type == "car"
        assert detection.position == (100.5, 200.3)
        assert detection.confidence == 0.92
    
    def test_vehicle_type_normalization(self):
        """Test that vehicle type is normalized to lowercase."""
        detection = VehicleDetection(
            vehicle_id="vehicle_123",
            vehicle_type="CAR",
            position=(100.5, 200.3),
            lane="north_lane_1",
            confidence=0.92,
            timestamp=datetime.now()
        )
        
        assert detection.vehicle_type == "car"
    
    def test_invalid_vehicle_type(self):
        """Test validation with invalid vehicle type."""
        with pytest.raises(ValueError, match="vehicle_type must be one of"):
            VehicleDetection(
                vehicle_id="vehicle_123",
                vehicle_type="airplane",
                position=(100.5, 200.3),
                lane="north_lane_1",
                confidence=0.92,
                timestamp=datetime.now()
            )
    
    def test_invalid_position(self):
        """Test validation with invalid position."""
        with pytest.raises(ValueError, match="position must be a tuple of two numbers"):
            VehicleDetection(
                vehicle_id="vehicle_123",
                vehicle_type="car",
                position=(100.5,),  # Only one coordinate
                lane="north_lane_1",
                confidence=0.92,
                timestamp=datetime.now()
            )
    
    def test_negative_position(self):
        """Test validation with negative position coordinates."""
        with pytest.raises(ValueError, match="position coordinates must be non-negative"):
            VehicleDetection(
                vehicle_id="vehicle_123",
                vehicle_type="car",
                position=(-10.0, 200.3),
                lane="north_lane_1",
                confidence=0.92,
                timestamp=datetime.now()
            )
    
    def test_invalid_confidence(self):
        """Test validation with invalid confidence value."""
        with pytest.raises(ValueError, match="confidence must be a number between 0.0 and 1.0"):
            VehicleDetection(
                vehicle_id="vehicle_123",
                vehicle_type="car",
                position=(100.5, 200.3),
                lane="north_lane_1",
                confidence=1.5,
                timestamp=datetime.now()
            )
    
    def test_helper_methods(self):
        """Test helper methods of VehicleDetection."""
        detection = VehicleDetection(
            vehicle_id="vehicle_123",
            vehicle_type="car",
            position=(3.0, 4.0),
            lane="north_lane_1",
            confidence=0.92,
            timestamp=datetime.now()
        )
        
        assert detection.is_high_confidence(0.9) == True
        assert detection.is_high_confidence(0.95) == False
        assert detection.get_distance_from_origin() == 5.0  # 3-4-5 triangle


class TestSignalAction:
    """Test cases for SignalAction model."""
    
    def test_valid_signal_action(self):
        """Test creating a valid signal action."""
        action = SignalAction(
            intersection_id="test_001",
            phase_adjustments={"north_south_green": 10, "east_west_green": -5},
            priority_direction="north",
            reasoning="Heavy traffic from north direction"
        )
        
        assert action.intersection_id == "test_001"
        assert action.phase_adjustments["north_south_green"] == 10
        assert action.priority_direction == "north"
        assert action.reasoning == "Heavy traffic from north direction"
    
    def test_empty_phase_adjustments(self):
        """Test validation with empty phase adjustments."""
        with pytest.raises(ValueError, match="phase_adjustments cannot be empty"):
            SignalAction(
                intersection_id="test_001",
                phase_adjustments={},
                reasoning="Test"
            )
    
    def test_invalid_adjustment_value(self):
        """Test validation with invalid adjustment value."""
        with pytest.raises(ValueError, match="adjustment for phase north_south_green must be an integer"):
            SignalAction(
                intersection_id="test_001",
                phase_adjustments={"north_south_green": 10.5},
                reasoning="Test"
            )
    
    def test_excessive_adjustment(self):
        """Test validation with excessive adjustment value."""
        with pytest.raises(ValueError, match="adjustment for phase north_south_green must be between -120 and 120 seconds"):
            SignalAction(
                intersection_id="test_001",
                phase_adjustments={"north_south_green": 150},
                reasoning="Test"
            )
    
    def test_invalid_priority_direction(self):
        """Test validation with invalid priority direction."""
        with pytest.raises(ValueError, match="priority_direction must be None or a non-empty string"):
            SignalAction(
                intersection_id="test_001",
                phase_adjustments={"north_south_green": 10},
                priority_direction="",
                reasoning="Test"
            )
    
    def test_helper_methods(self):
        """Test helper methods of SignalAction."""
        action = SignalAction(
            intersection_id="test_001",
            phase_adjustments={"north_south_green": 10, "east_west_green": -5, "yellow": 2},
            priority_direction="north",
            reasoning="Test"
        )
        
        assert action.get_total_adjustment() == 7  # 10 + (-5) + 2
        assert action.has_priority_direction() == True
        assert action.get_max_adjustment() == 10  # max(|10|, |-5|, |2|)
        
        action_no_priority = SignalAction(
            intersection_id="test_001",
            phase_adjustments={"north_south_green": 5},
            reasoning="Test"
        )
        
        assert action_no_priority.has_priority_direction() == False