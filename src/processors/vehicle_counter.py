"""Vehicle counting and classification system with lane-based tracking."""

from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque
import logging

from ..models.vehicle_detection import VehicleDetection

logger = logging.getLogger(__name__)


class VehicleTracker:
    """Tracks individual vehicles across frames for accurate counting."""
    
    def __init__(self, max_distance: float = 10.0, max_time_gap: int = 5):
        self.max_distance = max_distance  # Maximum distance for vehicle matching
        self.max_time_gap = max_time_gap  # Maximum time gap in seconds
        self.tracked_vehicles: Dict[str, VehicleDetection] = {}
        self.vehicle_paths: Dict[str, List[Tuple[float, float, datetime]]] = {}
        
    def update_tracks(self, detections: List[VehicleDetection]) -> List[VehicleDetection]:
        """Update vehicle tracks with new detections."""
        matched_detections = []
        unmatched_detections = list(detections)
        
        # Try to match new detections with existing tracks
        for vehicle_id, last_detection in list(self.tracked_vehicles.items()):
            best_match = None
            best_distance = float('inf')
            
            for detection in unmatched_detections:
                # Only match same vehicle type and lane
                if (detection.vehicle_type == last_detection.vehicle_type and 
                    detection.lane == last_detection.lane):
                    
                    distance = self._calculate_distance(
                        last_detection.position, detection.position
                    )
                    
                    time_diff = (detection.timestamp - last_detection.timestamp).total_seconds()
                    
                    if (distance < self.max_distance and 
                        time_diff < self.max_time_gap and 
                        distance < best_distance):
                        best_match = detection
                        best_distance = distance
            
            if best_match:
                # Update existing track
                updated_detection = VehicleDetection(
                    vehicle_id=vehicle_id,  # Keep original ID
                    vehicle_type=best_match.vehicle_type,
                    position=best_match.position,
                    lane=best_match.lane,
                    confidence=max(last_detection.confidence, best_match.confidence),
                    timestamp=best_match.timestamp
                )
                
                self.tracked_vehicles[vehicle_id] = updated_detection
                
                # Update path
                if vehicle_id not in self.vehicle_paths:
                    self.vehicle_paths[vehicle_id] = []
                self.vehicle_paths[vehicle_id].append(
                    (best_match.position[0], best_match.position[1], best_match.timestamp)
                )
                
                matched_detections.append(updated_detection)
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            self.tracked_vehicles[detection.vehicle_id] = detection
            self.vehicle_paths[detection.vehicle_id] = [
                (detection.position[0], detection.position[1], detection.timestamp)
            ]
            matched_detections.append(detection)
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        return matched_detections
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def _cleanup_old_tracks(self) -> None:
        """Remove tracks that haven't been updated recently."""
        current_time = datetime.now()
        old_tracks = []
        
        for vehicle_id, detection in self.tracked_vehicles.items():
            time_diff = (current_time - detection.timestamp).total_seconds()
            if time_diff > self.max_time_gap * 2:
                old_tracks.append(vehicle_id)
        
        for vehicle_id in old_tracks:
            del self.tracked_vehicles[vehicle_id]
            if vehicle_id in self.vehicle_paths:
                del self.vehicle_paths[vehicle_id]
    
    def get_vehicle_path(self, vehicle_id: str) -> List[Tuple[float, float, datetime]]:
        """Get the path history for a specific vehicle."""
        return self.vehicle_paths.get(vehicle_id, [])


class VehicleCounter:
    """Counts and classifies vehicles with lane-based tracking."""
    
    def __init__(self, intersection_config: Dict):
        self.intersection_config = intersection_config
        self.tracker = VehicleTracker()
        
        # Counting data structures
        self.lane_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.hourly_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_counts: Dict[str, int] = defaultdict(int)
        
        # Historical data (keep last 24 hours)
        self.historical_counts: deque = deque(maxlen=24 * 60)  # 24 hours of minute data
        
        # Counting zones for each lane
        self.counting_zones = self._setup_counting_zones()
        
        # Vehicles that have been counted (to avoid double counting)
        self.counted_vehicles: Set[str] = set()
        
        logger.info("VehicleCounter initialized")
    
    def _setup_counting_zones(self) -> Dict[str, Dict[str, float]]:
        """Setup counting zones for each lane."""
        zones = {}
        lanes = self.intersection_config.get('geometry', {}).get('lanes', {})
        
        for lane_name, lane_config in lanes.items():
            lane_length = lane_config.get('length', 100)
            
            # Define counting zone as the area 20-30 meters from intersection
            zones[lane_name] = {
                'min_distance': 20,
                'max_distance': 30,
                'lane_length': lane_length
            }
        
        return zones
    
    def process_detections(self, detections: List[VehicleDetection]) -> Dict[str, any]:
        """Process vehicle detections and update counts."""
        if not detections:
            return self._get_empty_counts()
        
        # Update vehicle tracks
        tracked_detections = self.tracker.update_tracks(detections)
        
        # Count vehicles in each lane
        current_counts = self._count_vehicles_by_lane(tracked_detections)
        
        # Update historical data
        self._update_historical_counts(current_counts)
        
        # Get classification data
        classification_data = self._classify_vehicles(tracked_detections)
        
        # Calculate flow rates
        flow_rates = self._calculate_flow_rates()
        
        result = {
            'timestamp': datetime.now(),
            'lane_counts': current_counts,
            'vehicle_classification': classification_data,
            'flow_rates': flow_rates,
            'total_vehicles': sum(current_counts.values()),
            'tracked_vehicles': len(self.tracker.tracked_vehicles)
        }
        
        logger.debug(f"Processed {len(detections)} detections, {len(tracked_detections)} tracked")
        return result
    
    def _count_vehicles_by_lane(self, detections: List[VehicleDetection]) -> Dict[str, int]:
        """Count vehicles in each lane."""
        lane_counts = defaultdict(int)
        
        for detection in detections:
            # Check if vehicle is in counting zone
            if self._is_in_counting_zone(detection):
                lane_counts[detection.lane] += 1
                
                # Mark as counted to avoid double counting
                if detection.vehicle_id not in self.counted_vehicles:
                    self.counted_vehicles.add(detection.vehicle_id)
                    self.total_counts[detection.vehicle_type] += 1
        
        return dict(lane_counts)
    
    def _is_in_counting_zone(self, detection: VehicleDetection) -> bool:
        """Check if a vehicle is in the counting zone."""
        # Extract lane direction from lane name
        lane_direction = detection.lane.split('_')[0]
        
        if lane_direction not in self.counting_zones:
            return True  # Default to counting if zone not defined
        
        zone = self.counting_zones[lane_direction]
        x, y = detection.position
        
        # Calculate distance from intersection center (0, 0)
        distance = (x ** 2 + y ** 2) ** 0.5
        
        return zone['min_distance'] <= distance <= zone['max_distance']
    
    def _classify_vehicles(self, detections: List[VehicleDetection]) -> Dict[str, Dict[str, int]]:
        """Classify vehicles by type and lane."""
        classification = defaultdict(lambda: defaultdict(int))
        
        for detection in detections:
            classification[detection.lane][detection.vehicle_type] += 1
        
        return dict(classification)
    
    def _calculate_flow_rates(self) -> Dict[str, float]:
        """Calculate vehicle flow rates (vehicles per minute) for each lane."""
        flow_rates = {}
        
        if len(self.historical_counts) < 2:
            return flow_rates
        
        # Calculate flow rate based on last 5 minutes of data
        recent_data = list(self.historical_counts)[-5:]
        
        for lane in self.lane_counts.keys():
            total_vehicles = sum(data.get(lane, 0) for data in recent_data)
            time_period = len(recent_data)  # minutes
            
            if time_period > 0:
                flow_rates[lane] = total_vehicles / time_period
            else:
                flow_rates[lane] = 0.0
        
        return flow_rates
    
    def _update_historical_counts(self, current_counts: Dict[str, int]) -> None:
        """Update historical counting data."""
        timestamp = datetime.now()
        
        # Store minute-level data
        self.historical_counts.append({
            'timestamp': timestamp,
            **current_counts
        })
        
        # Update hourly aggregates
        hour_key = timestamp.strftime('%Y-%m-%d-%H')
        for lane, count in current_counts.items():
            self.hourly_counts[hour_key][lane] += count
    
    def _get_empty_counts(self) -> Dict[str, any]:
        """Return empty count structure."""
        return {
            'timestamp': datetime.now(),
            'lane_counts': {},
            'vehicle_classification': {},
            'flow_rates': {},
            'total_vehicles': 0,
            'tracked_vehicles': 0
        }
    
    def get_lane_statistics(self, lane: str, hours: int = 1) -> Dict[str, any]:
        """Get statistics for a specific lane over time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        relevant_data = [
            data for data in self.historical_counts
            if data['timestamp'] >= cutoff_time and lane in data
        ]
        
        if not relevant_data:
            return {
                'lane': lane,
                'total_vehicles': 0,
                'average_flow_rate': 0.0,
                'peak_count': 0,
                'data_points': 0
            }
        
        counts = [data[lane] for data in relevant_data]
        
        return {
            'lane': lane,
            'total_vehicles': sum(counts),
            'average_flow_rate': sum(counts) / len(counts) if counts else 0.0,
            'peak_count': max(counts) if counts else 0,
            'data_points': len(counts)
        }
    
    def get_vehicle_type_distribution(self) -> Dict[str, float]:
        """Get distribution of vehicle types as percentages."""
        total = sum(self.total_counts.values())
        
        if total == 0:
            return {}
        
        return {
            vehicle_type: (count / total) * 100
            for vehicle_type, count in self.total_counts.items()
        }
    
    def reset_counts(self) -> None:
        """Reset all counting data."""
        self.lane_counts.clear()
        self.hourly_counts.clear()
        self.total_counts.clear()
        self.historical_counts.clear()
        self.counted_vehicles.clear()
        
        logger.info("Vehicle counts reset")
    
    def export_historical_data(self, hours: int = 24) -> List[Dict]:
        """Export historical counting data."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            data for data in self.historical_counts
            if data['timestamp'] >= cutoff_time
        ]