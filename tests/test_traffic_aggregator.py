"""Tests for traffic aggregator."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import statistics

from src.processors.traffic_aggregator import QueueAnalyzer, WaitTimeCalculator, TrafficAggregator
from src.models.vehicle_detection import VehicleDetection
from src.models.traffic_state import TrafficState


class TestQueueAnalyzer:
    """Test QueueAnalyzer class."""
    
    @pytest.fixture
    def intersection_config(self):
        """Sample intersection configuration."""
        return {
            'geometry': {
                'lanes': {
                    'north': {'count': 2, 'length': 100},
                    'south': {'count': 2, 'length': 100}
                }
            }
        }
    
    @pytest.fixture
    def analyzer(self, intersection_config):
        """Create queue analyzer instance."""
        return QueueAnalyzer(intersection_config)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.intersection_config is not None
        assert len(analyzer.queue_history) == 0
    
    def test_calculate_queue_length_empty(self, analyzer):
        """Test queue length calculation with no vehicles."""
        queue_length = analyzer.calculate_queue_length([], 'north')
        assert queue_length == 0.0
    
    def test_calculate_queue_length_with_vehicles(self, analyzer):
        """Test queue length calculation with vehicles."""
        detections = [
            VehicleDetection(
                vehicle_id='v1',
                vehicle_type='car',
                position=(0.0, 10.0),  # 10m from intersection
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='v2',
                vehicle_type='car',
                position=(0.0, 20.0),  # 20m from intersection
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='v3',
                vehicle_type='car',
                position=(0.0, 25.0),  # 25m from intersection
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            )
        ]
        
        queue_length = analyzer.calculate_queue_length(detections, 'north')
        
        # Queue length should be distance to furthest vehicle
        assert queue_length == 25.0
        
        # Check history is updated
        assert 'north' in analyzer.queue_history
        assert len(analyzer.queue_history['north']) == 1
    
    def test_calculate_queue_length_with_gap(self, analyzer):
        """Test queue length calculation with large gap between vehicles."""
        detections = [
            VehicleDetection(
                vehicle_id='v1',
                vehicle_type='car',
                position=(0.0, 10.0),  # 10m from intersection
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='v2',
                vehicle_type='car',
                position=(0.0, 30.0),  # 30m from intersection (large gap)
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            )
        ]
        
        queue_length = analyzer.calculate_queue_length(detections, 'north')
        
        # Should only count first vehicle due to gap
        assert queue_length == 10.0
    
    def test_distance_from_intersection(self, analyzer):
        """Test distance calculation from intersection."""
        distance = analyzer._distance_from_intersection((3.0, 4.0))
        assert distance == 5.0  # 3-4-5 triangle
        
        distance = analyzer._distance_from_intersection((0.0, 0.0))
        assert distance == 0.0
    
    def test_get_average_queue_length(self, analyzer):
        """Test getting average queue length."""
        # Add some history
        current_time = datetime.now()
        analyzer.queue_history['north'].append({
            'timestamp': current_time - timedelta(minutes=2),
            'length': 20.0,
            'vehicle_count': 3
        })
        analyzer.queue_history['north'].append({
            'timestamp': current_time - timedelta(minutes=1),
            'length': 30.0,
            'vehicle_count': 4
        })
        
        avg_length = analyzer.get_average_queue_length('north', minutes=5)
        assert avg_length == 25.0
    
    def test_get_average_queue_length_no_data(self, analyzer):
        """Test getting average with no data."""
        avg_length = analyzer.get_average_queue_length('unknown_lane', minutes=5)
        assert avg_length == 0.0


class TestWaitTimeCalculator:
    """Test WaitTimeCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create wait time calculator instance."""
        return WaitTimeCalculator()
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert len(calculator.vehicle_states) == 0
        assert len(calculator.wait_time_history) == 0
    
    def test_update_vehicle_states_new_vehicle(self, calculator):
        """Test updating states with new vehicle."""
        detection = VehicleDetection(
            vehicle_id='v1',
            vehicle_type='car',
            position=(10.0, 20.0),
            lane='north_lane_1',
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        calculator.update_vehicle_states([detection])
        
        assert 'v1' in calculator.vehicle_states
        state = calculator.vehicle_states['v1']
        assert state['lane'] == 'north_lane_1'
        assert state['last_position'] == (10.0, 20.0)
        assert state['is_moving'] == False
        assert state['total_wait_time'] == 0.0
    
    def test_update_vehicle_states_moving_vehicle(self, calculator):
        """Test updating states with moving vehicle."""
        timestamp1 = datetime.now()
        
        # First detection
        detection1 = VehicleDetection(
            vehicle_id='v1',
            vehicle_type='car',
            position=(10.0, 20.0),
            lane='north_lane_1',
            confidence=0.9,
            timestamp=timestamp1
        )
        
        calculator.update_vehicle_states([detection1])
        
        # Wait a bit to simulate time passing
        import time
        time.sleep(0.1)
        
        # Second detection (moved significantly) - use current time for realistic time diff
        detection2 = VehicleDetection(
            vehicle_id='v1',
            vehicle_type='car',
            position=(15.0, 25.0),  # Moved > 2 meters
            lane='north_lane_1',
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        # Manually set the last_seen time to make time_diff >= 5 seconds
        calculator.vehicle_states['v1']['last_seen'] = datetime.now() - timedelta(seconds=6)
        
        calculator.update_vehicle_states([detection2])
        
        state = calculator.vehicle_states['v1']
        assert state['is_moving'] == True
        assert state['stopped_time'] is None
    
    def test_calculate_distance(self, calculator):
        """Test distance calculation."""
        distance = calculator._calculate_distance((0.0, 0.0), (3.0, 4.0))
        assert distance == 5.0
    
    def test_cleanup_old_vehicles(self, calculator):
        """Test cleaning up old vehicles."""
        old_timestamp = datetime.now() - timedelta(minutes=2)
        
        # Add old vehicle
        calculator.vehicle_states['old_v'] = {
            'first_seen': old_timestamp,
            'last_position': (10.0, 20.0),
            'last_seen': old_timestamp,
            'lane': 'north_lane_1',
            'is_moving': False,
            'stopped_time': old_timestamp,
            'total_wait_time': 30.0
        }
        
        calculator._cleanup_old_vehicles()
        
        # Old vehicle should be removed
        assert 'old_v' not in calculator.vehicle_states
        
        # Wait time should be recorded in history
        assert 'north_lane_1' in calculator.wait_time_history
        assert len(calculator.wait_time_history['north_lane_1']) == 1
    
    def test_get_average_wait_time(self, calculator):
        """Test getting average wait time."""
        # Add wait time history
        current_time = datetime.now()
        calculator.wait_time_history['north'].append({
            'vehicle_id': 'v1',
            'wait_time': 30.0,
            'timestamp': current_time - timedelta(minutes=5)
        })
        calculator.wait_time_history['north'].append({
            'vehicle_id': 'v2',
            'wait_time': 50.0,
            'timestamp': current_time - timedelta(minutes=3)
        })
        
        avg_wait = calculator.get_average_wait_time('north', minutes=10)
        assert avg_wait == 40.0
    
    def test_get_average_wait_time_no_data(self, calculator):
        """Test getting average with no data."""
        avg_wait = calculator.get_average_wait_time('unknown_lane', minutes=10)
        assert avg_wait == 0.0
    
    def test_get_current_waiting_vehicles(self, calculator):
        """Test getting current waiting vehicles count."""
        current_time = datetime.now()
        
        # Add waiting vehicle
        calculator.vehicle_states['waiting_v'] = {
            'first_seen': current_time - timedelta(minutes=2),
            'last_position': (10.0, 20.0),
            'last_seen': current_time,
            'lane': 'north_lane_1',
            'is_moving': False,
            'stopped_time': current_time - timedelta(seconds=30),  # Waiting 30 seconds
            'total_wait_time': 0.0
        }
        
        # Add moving vehicle
        calculator.vehicle_states['moving_v'] = {
            'first_seen': current_time - timedelta(minutes=1),
            'last_position': (15.0, 25.0),
            'last_seen': current_time,
            'lane': 'north_lane_1',
            'is_moving': True,
            'stopped_time': None,
            'total_wait_time': 0.0
        }
        
        waiting_count = calculator.get_current_waiting_vehicles('north')
        assert waiting_count == 1  # Only the waiting vehicle


class TestTrafficAggregator:
    """Test TrafficAggregator class."""
    
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
    def aggregator(self, intersection_config):
        """Create traffic aggregator instance."""
        return TrafficAggregator('test_intersection', intersection_config)
    
    def test_aggregator_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.intersection_id == 'test_intersection'
        assert aggregator.intersection_config is not None
        assert aggregator.queue_analyzer is not None
        assert aggregator.wait_time_calculator is not None
        assert len(aggregator.traffic_history) == 0
        assert aggregator.current_signal_phase == "north_south_green"
    
    def test_aggregate_traffic_data(self, aggregator):
        """Test aggregating traffic data."""
        timestamp = datetime.now()
        detections = [
            VehicleDetection(
                vehicle_id='v1',
                vehicle_type='car',
                position=(0.0, 25.0),
                lane='north_lane_1',
                confidence=0.9,
                timestamp=timestamp
            ),
            VehicleDetection(
                vehicle_id='v2',
                vehicle_type='truck',
                position=(25.0, 0.0),
                lane='east_lane_1',
                confidence=0.8,
                timestamp=timestamp
            )
        ]
        
        vehicle_counts = {
            'north_lane_1': 1,
            'east_lane_1': 1
        }
        
        traffic_state = aggregator.aggregate_traffic_data(detections, vehicle_counts, timestamp)
        
        assert isinstance(traffic_state, TrafficState)
        assert traffic_state.intersection_id == 'test_intersection'
        assert traffic_state.timestamp == timestamp
        assert len(traffic_state.vehicle_counts) == 4  # All directions
        assert len(traffic_state.queue_lengths) == 4
        assert len(traffic_state.wait_times) == 4
        assert traffic_state.signal_phase == "north_south_green"
        assert 0.0 <= traffic_state.prediction_confidence <= 1.0
        
        # Check history is updated
        assert len(aggregator.traffic_history) == 1
    
    def test_calculate_prediction_confidence(self, aggregator):
        """Test prediction confidence calculation."""
        vehicle_counts = {'north': 5, 'south': 4, 'east': 6, 'west': 5}
        queue_lengths = {'north': 20.0, 'south': 15.0, 'east': 25.0, 'west': 18.0}
        wait_times = {'north': 30.0, 'south': 25.0, 'east': 35.0, 'west': 28.0}
        
        confidence = aggregator._calculate_prediction_confidence(
            vehicle_counts, queue_lengths, wait_times
        )
        
        assert 0.0 <= confidence <= 1.0
    
    def test_update_signal_phase(self, aggregator):
        """Test updating signal phase."""
        new_phase = "east_west_green"
        aggregator.update_signal_phase(new_phase)
        
        assert aggregator.current_signal_phase == new_phase
    
    def test_get_traffic_trends_no_data(self, aggregator):
        """Test getting trends with no data."""
        trends = aggregator.get_traffic_trends(hours=1)
        
        assert trends['trend_direction'] == 'stable'
        assert trends['average_vehicles'] == 0
        assert trends['peak_vehicles'] == 0
        assert trends['average_wait_time'] == 0.0
        assert trends['congestion_level'] == 'low'
    
    def test_get_traffic_trends_with_data(self, aggregator):
        """Test getting trends with historical data."""
        # Add some historical states
        base_time = datetime.now() - timedelta(minutes=30)
        
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i)
            vehicle_counts = {'north': 5 + i, 'south': 4, 'east': 3, 'west': 2}
            
            state = TrafficState(
                intersection_id='test',
                timestamp=timestamp,
                vehicle_counts=vehicle_counts,
                queue_lengths={'north': 20.0, 'south': 15.0, 'east': 10.0, 'west': 8.0},
                wait_times={'north': 30.0, 'south': 25.0, 'east': 20.0, 'west': 15.0},
                signal_phase='north_south_green',
                prediction_confidence=0.8
            )
            
            aggregator.traffic_history.append(state)
        
        trends = aggregator.get_traffic_trends(hours=1)
        
        assert trends['trend_direction'] in ['increasing', 'decreasing', 'stable']
        assert trends['average_vehicles'] > 0
        assert trends['peak_vehicles'] > 0
        assert trends['average_wait_time'] > 0
        assert trends['congestion_level'] in ['low', 'medium', 'high']
        assert trends['data_points'] == 20
    
    def test_get_performance_metrics_no_data(self, aggregator):
        """Test getting performance metrics with no data."""
        metrics = aggregator.get_performance_metrics()
        
        assert metrics['throughput'] == 0.0
        assert metrics['efficiency'] == 0.0
        assert metrics['average_delay'] == 0.0
        assert metrics['queue_clearance_time'] == 0.0
    
    def test_get_performance_metrics_with_data(self, aggregator):
        """Test getting performance metrics with data."""
        # Add some historical states
        base_time = datetime.now() - timedelta(minutes=60)
        
        for i in range(60):  # 60 minutes of data
            timestamp = base_time + timedelta(minutes=i)
            vehicle_counts = {'north': 5, 'south': 4, 'east': 3, 'west': 2}
            
            state = TrafficState(
                intersection_id='test',
                timestamp=timestamp,
                vehicle_counts=vehicle_counts,
                queue_lengths={'north': 20.0, 'south': 15.0, 'east': 10.0, 'west': 8.0},
                wait_times={'north': 30.0, 'south': 25.0, 'east': 20.0, 'west': 15.0},
                signal_phase='north_south_green',
                prediction_confidence=0.8
            )
            
            aggregator.traffic_history.append(state)
        
        metrics = aggregator.get_performance_metrics()
        
        assert metrics['throughput'] > 0.0
        assert 0.0 <= metrics['efficiency'] <= 1.0
        assert metrics['average_delay'] >= 0.0
        assert metrics['queue_clearance_time'] >= 0.0
    
    def test_export_traffic_data(self, aggregator):
        """Test exporting traffic data."""
        # Add some historical states
        recent_time = datetime.now() - timedelta(hours=1)
        old_time = datetime.now() - timedelta(hours=25)
        
        # Recent state
        recent_state = TrafficState(
            intersection_id='test',
            timestamp=recent_time,
            vehicle_counts={'north': 5, 'south': 4, 'east': 3, 'west': 2},
            queue_lengths={'north': 20.0, 'south': 15.0, 'east': 10.0, 'west': 8.0},
            wait_times={'north': 30.0, 'south': 25.0, 'east': 20.0, 'west': 15.0},
            signal_phase='north_south_green',
            prediction_confidence=0.8
        )
        
        # Old state
        old_state = TrafficState(
            intersection_id='test',
            timestamp=old_time,
            vehicle_counts={'north': 3, 'south': 2, 'east': 1, 'west': 1},
            queue_lengths={'north': 10.0, 'south': 8.0, 'east': 5.0, 'west': 4.0},
            wait_times={'north': 20.0, 'south': 15.0, 'east': 10.0, 'west': 8.0},
            signal_phase='east_west_green',
            prediction_confidence=0.7
        )
        
        aggregator.traffic_history.extend([recent_state, old_state])
        
        # Export last 24 hours
        exported = aggregator.export_traffic_data(hours=24)
        
        # Should only include recent data
        assert len(exported) == 1
        assert exported[0]['intersection_id'] == 'test'
        assert 'timestamp' in exported[0]
        assert 'vehicle_counts' in exported[0]
        assert 'total_vehicles' in exported[0]