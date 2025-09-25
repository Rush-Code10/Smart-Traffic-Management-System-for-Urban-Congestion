"""Tests for vehicle counter and tracker."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.processors.vehicle_counter import VehicleTracker, VehicleCounter
from src.models.vehicle_detection import VehicleDetection


class TestVehicleTracker:
    """Test VehicleTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create vehicle tracker instance."""
        return VehicleTracker(max_distance=10.0, max_time_gap=5)
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.max_distance == 10.0
        assert tracker.max_time_gap == 5
        assert len(tracker.tracked_vehicles) == 0
        assert len(tracker.vehicle_paths) == 0
    
    def test_update_tracks_new_vehicles(self, tracker):
        """Test updating tracks with new vehicles."""
        timestamp = datetime.now()
        detections = [
            VehicleDetection(
                vehicle_id='vehicle_001',
                vehicle_type='car',
                position=(10.0, 20.0),
                lane='north_lane_1',
                confidence=0.9,
                timestamp=timestamp
            ),
            VehicleDetection(
                vehicle_id='vehicle_002',
                vehicle_type='truck',
                position=(15.0, 25.0),
                lane='south_lane_1',
                confidence=0.8,
                timestamp=timestamp
            )
        ]
        
        tracked = tracker.update_tracks(detections)
        
        assert len(tracked) == 2
        assert len(tracker.tracked_vehicles) == 2
        assert 'vehicle_001' in tracker.tracked_vehicles
        assert 'vehicle_002' in tracker.tracked_vehicles
        
        # Check paths are created
        assert 'vehicle_001' in tracker.vehicle_paths
        assert 'vehicle_002' in tracker.vehicle_paths
        assert len(tracker.vehicle_paths['vehicle_001']) == 1
    
    def test_update_tracks_existing_vehicles(self, tracker):
        """Test updating tracks with existing vehicles."""
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(seconds=2)
        
        # First detection
        detection1 = VehicleDetection(
            vehicle_id='vehicle_001',
            vehicle_type='car',
            position=(10.0, 20.0),
            lane='north_lane_1',
            confidence=0.9,
            timestamp=timestamp1
        )
        
        tracker.update_tracks([detection1])
        
        # Second detection (same vehicle, slightly moved)
        detection2 = VehicleDetection(
            vehicle_id='vehicle_003',  # Different ID but should match
            vehicle_type='car',
            position=(12.0, 22.0),  # Moved slightly
            lane='north_lane_1',
            confidence=0.85,
            timestamp=timestamp2
        )
        
        tracked = tracker.update_tracks([detection2])
        
        # Should still have only one tracked vehicle
        assert len(tracker.tracked_vehicles) == 1
        
        # Should maintain original vehicle ID
        original_id = list(tracker.tracked_vehicles.keys())[0]
        assert original_id == 'vehicle_001'
        
        # Path should have two points
        assert len(tracker.vehicle_paths[original_id]) == 2
    
    def test_calculate_distance(self, tracker):
        """Test distance calculation."""
        pos1 = (0.0, 0.0)
        pos2 = (3.0, 4.0)
        
        distance = tracker._calculate_distance(pos1, pos2)
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_cleanup_old_tracks(self, tracker):
        """Test cleaning up old tracks."""
        old_timestamp = datetime.now() - timedelta(seconds=20)
        
        # Add an old detection
        detection = VehicleDetection(
            vehicle_id='old_vehicle',
            vehicle_type='car',
            position=(10.0, 20.0),
            lane='north_lane_1',
            confidence=0.9,
            timestamp=old_timestamp
        )
        
        tracker.tracked_vehicles['old_vehicle'] = detection
        tracker.vehicle_paths['old_vehicle'] = [(10.0, 20.0, old_timestamp)]
        
        # Cleanup should remove old track
        tracker._cleanup_old_tracks()
        
        assert 'old_vehicle' not in tracker.tracked_vehicles
        assert 'old_vehicle' not in tracker.vehicle_paths
    
    def test_get_vehicle_path(self, tracker):
        """Test getting vehicle path."""
        vehicle_id = 'test_vehicle'
        path = [(10.0, 20.0, datetime.now())]
        tracker.vehicle_paths[vehicle_id] = path
        
        retrieved_path = tracker.get_vehicle_path(vehicle_id)
        assert retrieved_path == path
        
        # Test non-existent vehicle
        empty_path = tracker.get_vehicle_path('non_existent')
        assert empty_path == []


class TestVehicleCounter:
    """Test VehicleCounter class."""
    
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
    def counter(self, intersection_config):
        """Create vehicle counter instance."""
        return VehicleCounter(intersection_config)
    
    def test_counter_initialization(self, counter):
        """Test counter initialization."""
        assert counter.intersection_config is not None
        assert counter.tracker is not None
        assert len(counter.counting_zones) == 4
        assert len(counter.counted_vehicles) == 0
        
        # Check counting zones setup
        assert 'north' in counter.counting_zones
        assert counter.counting_zones['north']['min_distance'] == 20
        assert counter.counting_zones['north']['max_distance'] == 30
    
    def test_process_detections_empty(self, counter):
        """Test processing empty detections."""
        result = counter.process_detections([])
        
        assert result['total_vehicles'] == 0
        assert result['lane_counts'] == {}
        assert result['vehicle_classification'] == {}
        assert result['tracked_vehicles'] == 0
    
    def test_process_detections_with_vehicles(self, counter):
        """Test processing detections with vehicles."""
        timestamp = datetime.now()
        detections = [
            VehicleDetection(
                vehicle_id='vehicle_001',
                vehicle_type='car',
                position=(25.0, 0.0),  # In counting zone
                lane='north_lane_1',
                confidence=0.9,
                timestamp=timestamp
            ),
            VehicleDetection(
                vehicle_id='vehicle_002',
                vehicle_type='truck',
                position=(0.0, 25.0),  # In counting zone
                lane='east_lane_1',
                confidence=0.8,
                timestamp=timestamp
            )
        ]
        
        result = counter.process_detections(detections)
        
        assert result['total_vehicles'] > 0
        assert isinstance(result['lane_counts'], dict)
        assert isinstance(result['vehicle_classification'], dict)
        assert result['tracked_vehicles'] > 0
        assert 'timestamp' in result
    
    def test_is_in_counting_zone(self, counter):
        """Test counting zone detection."""
        # Vehicle in counting zone (distance 25 from center)
        detection_in_zone = VehicleDetection(
            vehicle_id='test1',
            vehicle_type='car',
            position=(25.0, 0.0),
            lane='north_lane_1',
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        assert counter._is_in_counting_zone(detection_in_zone) == True
        
        # Vehicle outside counting zone (distance 50 from center)
        detection_out_zone = VehicleDetection(
            vehicle_id='test2',
            vehicle_type='car',
            position=(50.0, 0.0),
            lane='north_lane_1',
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        assert counter._is_in_counting_zone(detection_out_zone) == False
        
        # Vehicle too close to intersection (distance 10 from center)
        detection_too_close = VehicleDetection(
            vehicle_id='test3',
            vehicle_type='car',
            position=(10.0, 0.0),
            lane='north_lane_1',
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        assert counter._is_in_counting_zone(detection_too_close) == False
    
    def test_count_vehicles_by_lane(self, counter):
        """Test counting vehicles by lane."""
        detections = [
            VehicleDetection(
                vehicle_id='v1',
                vehicle_type='car',
                position=(25.0, 0.0),
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='v2',
                vehicle_type='car',
                position=(25.0, 0.0),
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='v3',
                vehicle_type='truck',
                position=(0.0, 25.0),
                lane='east_lane_1',
                confidence=0.8,
                timestamp=datetime.now()
            )
        ]
        
        counts = counter._count_vehicles_by_lane(detections)
        
        assert counts['north_lane_1'] == 2
        assert counts['east_lane_1'] == 1
    
    def test_classify_vehicles(self, counter):
        """Test vehicle classification."""
        detections = [
            VehicleDetection(
                vehicle_id='v1',
                vehicle_type='car',
                position=(25.0, 0.0),
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='v2',
                vehicle_type='truck',
                position=(25.0, 0.0),
                lane='north_lane_1',
                confidence=0.8,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='v3',
                vehicle_type='car',
                position=(0.0, 25.0),
                lane='east_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            )
        ]
        
        classification = counter._classify_vehicles(detections)
        
        assert classification['north_lane_1']['car'] == 1
        assert classification['north_lane_1']['truck'] == 1
        assert classification['east_lane_1']['car'] == 1
    
    def test_get_lane_statistics(self, counter):
        """Test getting lane statistics."""
        # Add some historical data
        counter.historical_counts.append({
            'timestamp': datetime.now() - timedelta(minutes=30),
            'north_lane_1': 5,
            'east_lane_1': 3
        })
        counter.historical_counts.append({
            'timestamp': datetime.now() - timedelta(minutes=15),
            'north_lane_1': 8,
            'east_lane_1': 4
        })
        
        stats = counter.get_lane_statistics('north_lane_1', hours=1)
        
        assert stats['lane'] == 'north_lane_1'
        assert stats['total_vehicles'] == 13
        assert stats['average_flow_rate'] == 6.5
        assert stats['peak_count'] == 8
        assert stats['data_points'] == 2
    
    def test_get_lane_statistics_no_data(self, counter):
        """Test getting statistics for lane with no data."""
        stats = counter.get_lane_statistics('unknown_lane', hours=1)
        
        assert stats['lane'] == 'unknown_lane'
        assert stats['total_vehicles'] == 0
        assert stats['average_flow_rate'] == 0.0
        assert stats['peak_count'] == 0
        assert stats['data_points'] == 0
    
    def test_get_vehicle_type_distribution(self, counter):
        """Test getting vehicle type distribution."""
        # Add some counts
        counter.total_counts['car'] = 70
        counter.total_counts['truck'] = 20
        counter.total_counts['bus'] = 10
        
        distribution = counter.get_vehicle_type_distribution()
        
        assert distribution['car'] == 70.0
        assert distribution['truck'] == 20.0
        assert distribution['bus'] == 10.0
    
    def test_get_vehicle_type_distribution_empty(self, counter):
        """Test getting distribution with no data."""
        distribution = counter.get_vehicle_type_distribution()
        assert distribution == {}
    
    def test_reset_counts(self, counter):
        """Test resetting all counts."""
        # Add some data
        counter.lane_counts['north_lane_1'] = 5
        counter.total_counts['car'] = 10
        counter.counted_vehicles.add('vehicle_001')
        counter.historical_counts.append({'test': 'data'})
        
        counter.reset_counts()
        
        assert len(counter.lane_counts) == 0
        assert len(counter.total_counts) == 0
        assert len(counter.counted_vehicles) == 0
        assert len(counter.historical_counts) == 0
    
    def test_export_historical_data(self, counter):
        """Test exporting historical data."""
        # Add historical data
        recent_time = datetime.now() - timedelta(minutes=30)
        old_time = datetime.now() - timedelta(hours=25)
        
        counter.historical_counts.append({
            'timestamp': recent_time,
            'north_lane_1': 5
        })
        counter.historical_counts.append({
            'timestamp': old_time,
            'north_lane_1': 3
        })
        
        # Export last 24 hours
        exported = counter.export_historical_data(hours=24)
        
        # Should only include recent data
        assert len(exported) == 1
        assert exported[0]['timestamp'] == recent_time
    
    def test_calculate_flow_rates(self, counter):
        """Test flow rate calculation."""
        # Add historical data
        for i in range(5):
            counter.historical_counts.append({
                'timestamp': datetime.now() - timedelta(minutes=i),
                'north_lane_1': 2,
                'east_lane_1': 3
            })
        
        flow_rates = counter._calculate_flow_rates()
        
        # Should calculate flow rates for lanes with data
        if 'north_lane_1' in flow_rates:
            assert flow_rates['north_lane_1'] == 2.0  # 2 vehicles per minute
        if 'east_lane_1' in flow_rates:
            assert flow_rates['east_lane_1'] == 3.0  # 3 vehicles per minute