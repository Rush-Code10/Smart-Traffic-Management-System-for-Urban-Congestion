"""Traffic data aggregator that computes queue lengths and wait times."""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import statistics
import logging

from ..models.vehicle_detection import VehicleDetection
from ..models.traffic_state import TrafficState

logger = logging.getLogger(__name__)


class QueueAnalyzer:
    """Analyzes vehicle queues and calculates queue lengths."""
    
    def __init__(self, intersection_config: Dict):
        self.intersection_config = intersection_config
        self.queue_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 5 minutes at 5-second intervals
        
    def calculate_queue_length(self, detections: List[VehicleDetection], 
                             lane: str) -> float:
        """Calculate queue length for a specific lane."""
        lane_detections = [d for d in detections if d.lane.startswith(lane)]
        
        if not lane_detections:
            return 0.0
        
        # Sort vehicles by distance from intersection (closer = smaller distance)
        lane_detections.sort(key=lambda d: self._distance_from_intersection(d.position))
        
        # Find the queue by looking for clusters of vehicles
        queue_length = 0.0
        vehicle_positions = [self._distance_from_intersection(d.position) for d in lane_detections]
        
        if not vehicle_positions:
            return 0.0
        
        # Simple queue detection: vehicles within 50 meters of intersection
        # and with gaps less than 10 meters between them
        queue_vehicles = []
        for i, distance in enumerate(vehicle_positions):
            if distance <= 50:  # Within 50m of intersection
                if not queue_vehicles or (distance - queue_vehicles[-1]) <= 10:
                    queue_vehicles.append(distance)
                else:
                    break  # Gap too large, end of queue
        
        if queue_vehicles:
            # Queue length is from intersection to last vehicle in queue
            queue_length = max(queue_vehicles)
        
        # Store in history
        self.queue_history[lane].append({
            'timestamp': datetime.now(),
            'length': queue_length,
            'vehicle_count': len(queue_vehicles)
        })
        
        return queue_length
    
    def _distance_from_intersection(self, position: Tuple[float, float]) -> float:
        """Calculate distance from intersection center."""
        x, y = position
        return (x ** 2 + y ** 2) ** 0.5
    
    def get_average_queue_length(self, lane: str, minutes: int = 5) -> float:
        """Get average queue length over specified time period."""
        if lane not in self.queue_history:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [
            entry['length'] for entry in self.queue_history[lane]
            if entry['timestamp'] >= cutoff_time
        ]
        
        return statistics.mean(recent_data) if recent_data else 0.0


class WaitTimeCalculator:
    """Calculates vehicle wait times based on movement patterns."""
    
    def __init__(self):
        self.vehicle_states: Dict[str, Dict] = {}
        self.wait_time_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def update_vehicle_states(self, detections: List[VehicleDetection]) -> None:
        """Update vehicle states for wait time calculation."""
        current_time = datetime.now()
        
        for detection in detections:
            vehicle_id = detection.vehicle_id
            
            if vehicle_id not in self.vehicle_states:
                # New vehicle
                self.vehicle_states[vehicle_id] = {
                    'first_seen': current_time,
                    'last_position': detection.position,
                    'last_seen': current_time,
                    'lane': detection.lane,
                    'is_moving': False,
                    'stopped_time': None,
                    'total_wait_time': 0.0
                }
            else:
                # Existing vehicle
                state = self.vehicle_states[vehicle_id]
                
                # Calculate movement
                distance_moved = self._calculate_distance(
                    state['last_position'], detection.position
                )
                
                time_diff = (current_time - state['last_seen']).total_seconds()
                
                # Determine if vehicle is moving (threshold: 2 meters in 5 seconds)
                is_moving = distance_moved > 2.0 and time_diff >= 5.0
                
                if not is_moving and state['is_moving']:
                    # Vehicle just stopped
                    state['stopped_time'] = current_time
                    state['is_moving'] = False
                elif is_moving and not state['is_moving']:
                    # Vehicle started moving
                    if state['stopped_time']:
                        wait_time = (current_time - state['stopped_time']).total_seconds()
                        state['total_wait_time'] += wait_time
                    state['is_moving'] = True
                    state['stopped_time'] = None
                
                # Update state
                state['last_position'] = detection.position
                state['last_seen'] = current_time
        
        # Clean up old vehicles
        self._cleanup_old_vehicles()
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def _cleanup_old_vehicles(self) -> None:
        """Remove vehicles that haven't been seen recently."""
        current_time = datetime.now()
        old_vehicles = []
        
        for vehicle_id, state in self.vehicle_states.items():
            time_since_seen = (current_time - state['last_seen']).total_seconds()
            
            if time_since_seen > 60:  # 1 minute
                # Calculate final wait time if vehicle was stopped
                if not state['is_moving'] and state['stopped_time']:
                    final_wait = (current_time - state['stopped_time']).total_seconds()
                    state['total_wait_time'] += final_wait
                
                # Store wait time in history
                if state['total_wait_time'] > 0:
                    self.wait_time_history[state['lane']].append({
                        'vehicle_id': vehicle_id,
                        'wait_time': state['total_wait_time'],
                        'timestamp': current_time
                    })
                
                old_vehicles.append(vehicle_id)
        
        for vehicle_id in old_vehicles:
            del self.vehicle_states[vehicle_id]
    
    def get_average_wait_time(self, lane: str, minutes: int = 10) -> float:
        """Get average wait time for a lane over specified period."""
        if lane not in self.wait_time_history:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_waits = [
            entry['wait_time'] for entry in self.wait_time_history[lane]
            if entry['timestamp'] >= cutoff_time
        ]
        
        return statistics.mean(recent_waits) if recent_waits else 0.0
    
    def get_current_waiting_vehicles(self, lane: str) -> int:
        """Get count of currently waiting vehicles in a lane."""
        current_time = datetime.now()
        waiting_count = 0
        
        for state in self.vehicle_states.values():
            if (state['lane'].startswith(lane) and 
                not state['is_moving'] and 
                state['stopped_time'] and
                (current_time - state['stopped_time']).total_seconds() > 10):  # Waiting > 10 seconds
                waiting_count += 1
        
        return waiting_count


class TrafficAggregator:
    """Aggregates traffic data and computes comprehensive traffic state."""
    
    def __init__(self, intersection_id: str, intersection_config: Dict):
        self.intersection_id = intersection_id
        self.intersection_config = intersection_config
        
        # Initialize analyzers
        self.queue_analyzer = QueueAnalyzer(intersection_config)
        self.wait_time_calculator = WaitTimeCalculator()
        
        # Historical data storage
        self.traffic_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Current signal state (would be updated by signal controller)
        self.current_signal_phase = "north_south_green"
        
        logger.info(f"TrafficAggregator initialized for intersection {intersection_id}")
    
    def aggregate_traffic_data(self, detections: List[VehicleDetection],
                             vehicle_counts: Dict[str, int],
                             timestamp: Optional[datetime] = None) -> TrafficState:
        """Aggregate all traffic data into a TrafficState object."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update wait time calculations
        self.wait_time_calculator.update_vehicle_states(detections)
        
        # Get lane directions from intersection config
        lanes = self.intersection_config.get('geometry', {}).get('lanes', {})
        lane_directions = list(lanes.keys())
        
        # Calculate queue lengths for each direction
        queue_lengths = {}
        for direction in lane_directions:
            queue_length = self.queue_analyzer.calculate_queue_length(detections, direction)
            queue_lengths[direction] = queue_length
        
        # Calculate wait times for each direction
        wait_times = {}
        for direction in lane_directions:
            wait_time = self.wait_time_calculator.get_average_wait_time(direction)
            wait_times[direction] = wait_time
        
        # Ensure vehicle_counts has all directions
        complete_vehicle_counts = {}
        for direction in lane_directions:
            direction_count = sum(
                count for lane, count in vehicle_counts.items()
                if lane.startswith(direction)
            )
            complete_vehicle_counts[direction] = direction_count
        
        # Calculate prediction confidence (simplified)
        prediction_confidence = self._calculate_prediction_confidence(
            complete_vehicle_counts, queue_lengths, wait_times
        )
        
        # Create traffic state
        traffic_state = TrafficState(
            intersection_id=self.intersection_id,
            timestamp=timestamp,
            vehicle_counts=complete_vehicle_counts,
            queue_lengths=queue_lengths,
            wait_times=wait_times,
            signal_phase=self.current_signal_phase,
            prediction_confidence=prediction_confidence
        )
        
        # Store in history
        self.traffic_history.append(traffic_state)
        
        logger.debug(f"Aggregated traffic data: {len(detections)} detections, "
                    f"{sum(complete_vehicle_counts.values())} total vehicles")
        
        return traffic_state
    
    def _calculate_prediction_confidence(self, vehicle_counts: Dict[str, int],
                                       queue_lengths: Dict[str, float],
                                       wait_times: Dict[str, float]) -> float:
        """Calculate confidence in traffic predictions based on data quality."""
        confidence_factors = []
        
        # Factor 1: Data completeness (all directions have data)
        total_directions = len(self.intersection_config.get('geometry', {}).get('lanes', {}))
        directions_with_data = sum(1 for count in vehicle_counts.values() if count > 0)
        
        if total_directions > 0:
            completeness = directions_with_data / total_directions
            confidence_factors.append(completeness)
        
        # Factor 2: Data consistency (similar patterns across directions)
        if len(vehicle_counts) > 1:
            counts = list(vehicle_counts.values())
            if max(counts) > 0:
                consistency = 1.0 - (statistics.stdev(counts) / max(counts))
                confidence_factors.append(max(0.0, consistency))
        
        # Factor 3: Historical data availability
        history_factor = min(1.0, len(self.traffic_history) / 60)  # Full confidence after 1 hour
        confidence_factors.append(history_factor)
        
        # Factor 4: Queue length reasonableness (not too extreme)
        if queue_lengths:
            max_queue = max(queue_lengths.values())
            queue_factor = 1.0 if max_queue < 100 else max(0.5, 100 / max_queue)
            confidence_factors.append(queue_factor)
        
        # Calculate overall confidence
        if confidence_factors:
            return statistics.mean(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def update_signal_phase(self, new_phase: str) -> None:
        """Update current signal phase."""
        self.current_signal_phase = new_phase
        logger.debug(f"Signal phase updated to: {new_phase}")
    
    def get_traffic_trends(self, hours: int = 1) -> Dict[str, any]:
        """Get traffic trends over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_states = [
            state for state in self.traffic_history
            if state.timestamp >= cutoff_time
        ]
        
        if not recent_states:
            return {
                'trend_direction': 'stable',
                'average_vehicles': 0,
                'peak_vehicles': 0,
                'average_wait_time': 0.0,
                'congestion_level': 'low'
            }
        
        # Calculate trends
        total_vehicles = [state.get_total_vehicles() for state in recent_states]
        avg_wait_times = [state.get_average_wait_time() for state in recent_states]
        
        # Determine trend direction
        if len(total_vehicles) >= 2:
            recent_avg = statistics.mean(total_vehicles[-10:])  # Last 10 data points
            earlier_avg = statistics.mean(total_vehicles[:10])   # First 10 data points
            
            if recent_avg > earlier_avg * 1.1:
                trend_direction = 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
        
        # Determine congestion level
        avg_vehicles = statistics.mean(total_vehicles)
        avg_wait = statistics.mean(avg_wait_times) if avg_wait_times else 0.0
        
        if avg_vehicles > 20 or avg_wait > 60:
            congestion_level = 'high'
        elif avg_vehicles > 10 or avg_wait > 30:
            congestion_level = 'medium'
        else:
            congestion_level = 'low'
        
        return {
            'trend_direction': trend_direction,
            'average_vehicles': avg_vehicles,
            'peak_vehicles': max(total_vehicles) if total_vehicles else 0,
            'average_wait_time': avg_wait,
            'congestion_level': congestion_level,
            'data_points': len(recent_states)
        }
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """Get performance metrics for the intersection."""
        if not self.traffic_history:
            return {
                'throughput': 0.0,
                'efficiency': 0.0,
                'average_delay': 0.0,
                'queue_clearance_time': 0.0
            }
        
        recent_states = list(self.traffic_history)[-60:]  # Last hour
        
        # Calculate throughput (vehicles per hour)
        total_vehicles = sum(state.get_total_vehicles() for state in recent_states)
        throughput = total_vehicles * (60 / len(recent_states)) if recent_states else 0.0
        
        # Calculate efficiency (inverse of average wait time)
        avg_wait_times = [state.get_average_wait_time() for state in recent_states]
        avg_wait = statistics.mean(avg_wait_times) if avg_wait_times else 0.0
        efficiency = 1.0 / (1.0 + avg_wait / 60.0)  # Normalized efficiency
        
        # Calculate average delay
        average_delay = avg_wait
        
        # Estimate queue clearance time (simplified)
        queue_clearance_time = avg_wait * 1.5  # Rough estimate
        
        return {
            'throughput': throughput,
            'efficiency': efficiency,
            'average_delay': average_delay,
            'queue_clearance_time': queue_clearance_time
        }
    
    def export_traffic_data(self, hours: int = 24) -> List[Dict]:
        """Export traffic data for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        exported_data = []
        for state in self.traffic_history:
            if state.timestamp >= cutoff_time:
                exported_data.append({
                    'timestamp': state.timestamp.isoformat(),
                    'intersection_id': state.intersection_id,
                    'vehicle_counts': state.vehicle_counts,
                    'queue_lengths': state.queue_lengths,
                    'wait_times': state.wait_times,
                    'signal_phase': state.signal_phase,
                    'prediction_confidence': state.prediction_confidence,
                    'total_vehicles': state.get_total_vehicles(),
                    'total_queue_length': state.get_total_queue_length(),
                    'average_wait_time': state.get_average_wait_time()
                })
        
        return exported_data