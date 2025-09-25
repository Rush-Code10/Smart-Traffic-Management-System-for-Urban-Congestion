"""Synthetic traffic data generator for realistic vehicle flows."""

import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from ..models.vehicle_detection import VehicleDetection

logger = logging.getLogger(__name__)


@dataclass
class TrafficPattern:
    """Defines traffic patterns for different times and conditions."""
    
    base_flow_rate: float  # vehicles per minute per lane
    peak_multiplier: float = 1.5
    vehicle_type_distribution: Dict[str, float] = None
    speed_range: Tuple[float, float] = (20.0, 60.0)  # km/h
    
    def __post_init__(self):
        if self.vehicle_type_distribution is None:
            self.vehicle_type_distribution = {
                'car': 0.75,
                'truck': 0.15,
                'bus': 0.05,
                'motorcycle': 0.05
            }


class TrafficSimulator:
    """Generates synthetic traffic data with realistic patterns."""
    
    def __init__(self, intersection_config: Dict):
        self.intersection_config = intersection_config
        self.vehicle_counter = 0
        self.active_vehicles: Dict[str, VehicleDetection] = {}
        
        # Traffic patterns for different times
        self.patterns = {
            'morning_rush': TrafficPattern(
                base_flow_rate=15.0,
                peak_multiplier=2.0,
                vehicle_type_distribution={'car': 0.8, 'truck': 0.1, 'bus': 0.05, 'motorcycle': 0.05}
            ),
            'evening_rush': TrafficPattern(
                base_flow_rate=12.0,
                peak_multiplier=1.8,
                vehicle_type_distribution={'car': 0.85, 'truck': 0.08, 'bus': 0.04, 'motorcycle': 0.03}
            ),
            'midday': TrafficPattern(
                base_flow_rate=8.0,
                peak_multiplier=1.2,
                vehicle_type_distribution={'car': 0.7, 'truck': 0.2, 'bus': 0.06, 'motorcycle': 0.04}
            ),
            'night': TrafficPattern(
                base_flow_rate=3.0,
                peak_multiplier=1.0,
                vehicle_type_distribution={'car': 0.9, 'truck': 0.05, 'bus': 0.02, 'motorcycle': 0.03}
            )
        }
        
        logger.info(f"TrafficSimulator initialized for intersection")
    
    def get_current_pattern(self, timestamp: datetime) -> TrafficPattern:
        """Get traffic pattern based on time of day."""
        hour = timestamp.hour
        
        if 7 <= hour <= 9:
            return self.patterns['morning_rush']
        elif 17 <= hour <= 19:
            return self.patterns['evening_rush']
        elif 10 <= hour <= 16:
            return self.patterns['midday']
        else:
            return self.patterns['night']
    
    def _generate_vehicle_type(self, pattern: TrafficPattern) -> str:
        """Generate vehicle type based on distribution."""
        rand = random.random()
        cumulative = 0.0
        
        for vehicle_type, probability in pattern.vehicle_type_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return vehicle_type
        
        return 'car'  # fallback
    
    def _generate_vehicle_position(self, lane: str, lane_config: Dict) -> Tuple[float, float]:
        """Generate realistic vehicle position within lane."""
        lane_length = lane_config.get('length', 100)
        
        # Position vehicles randomly along the lane with some clustering near signals
        if random.random() < 0.3:  # 30% chance of being near signal (queue)
            distance_from_signal = random.uniform(0, 30)
        else:
            distance_from_signal = random.uniform(30, lane_length)
        
        # Convert to x, y coordinates based on direction
        if lane == 'north':
            return (0, distance_from_signal)
        elif lane == 'south':
            return (0, -distance_from_signal)
        elif lane == 'east':
            return (distance_from_signal, 0)
        elif lane == 'west':
            return (-distance_from_signal, 0)
        else:
            return (0, 0)
    
    def generate_vehicle_detections(self, timestamp: datetime, 
                                  duration_seconds: int = 5) -> List[VehicleDetection]:
        """Generate vehicle detections for a time period."""
        pattern = self.get_current_pattern(timestamp)
        detections = []
        
        # Get lanes from intersection config
        lanes = self.intersection_config.get('geometry', {}).get('lanes', {})
        
        for lane_name, lane_config in lanes.items():
            lane_count = lane_config.get('count', 1)
            
            # Calculate vehicles to generate based on flow rate
            flow_rate = pattern.base_flow_rate * pattern.peak_multiplier
            vehicles_per_lane = max(1, int((flow_rate * duration_seconds / 60) * lane_count))
            
            # Add some randomness
            vehicles_per_lane = max(0, vehicles_per_lane + random.randint(-2, 3))
            
            for _ in range(vehicles_per_lane):
                self.vehicle_counter += 1
                vehicle_id = f"vehicle_{self.vehicle_counter:06d}"
                
                detection = VehicleDetection(
                    vehicle_id=vehicle_id,
                    vehicle_type=self._generate_vehicle_type(pattern),
                    position=self._generate_vehicle_position(lane_name, lane_config),
                    lane=f"{lane_name}_lane_{random.randint(1, lane_count)}",
                    confidence=random.uniform(0.85, 0.98),
                    timestamp=timestamp + timedelta(seconds=random.uniform(0, duration_seconds))
                )
                
                detections.append(detection)
                self.active_vehicles[vehicle_id] = detection
        
        logger.debug(f"Generated {len(detections)} vehicle detections")
        return detections
    
    def simulate_traffic_event(self, event_type: str, timestamp: datetime) -> List[VehicleDetection]:
        """Simulate special traffic events (accidents, construction, etc.)."""
        detections = []
        
        if event_type == 'accident':
            # Reduce traffic flow and create backup
            affected_lane = random.choice(['north', 'south', 'east', 'west'])
            lanes = self.intersection_config.get('geometry', {}).get('lanes', {})
            
            if affected_lane in lanes:
                lane_config = lanes[affected_lane]
                # Generate fewer vehicles but clustered
                for i in range(random.randint(5, 15)):
                    self.vehicle_counter += 1
                    vehicle_id = f"accident_vehicle_{self.vehicle_counter:06d}"
                    
                    # Cluster vehicles behind accident
                    base_pos = self._generate_vehicle_position(affected_lane, lane_config)
                    clustered_pos = (
                        base_pos[0] + random.uniform(-5, 5),
                        base_pos[1] + random.uniform(-20, -5)
                    )
                    
                    detection = VehicleDetection(
                        vehicle_id=vehicle_id,
                        vehicle_type=self._generate_vehicle_type(self.patterns['midday']),
                        position=clustered_pos,
                        lane=f"{affected_lane}_lane_1",
                        confidence=random.uniform(0.8, 0.95),
                        timestamp=timestamp
                    )
                    
                    detections.append(detection)
        
        elif event_type == 'rush_hour_spike':
            # Temporarily increase traffic in all directions
            for lane_name, lane_config in self.intersection_config.get('geometry', {}).get('lanes', {}).items():
                for _ in range(random.randint(8, 15)):
                    self.vehicle_counter += 1
                    vehicle_id = f"rush_vehicle_{self.vehicle_counter:06d}"
                    
                    detection = VehicleDetection(
                        vehicle_id=vehicle_id,
                        vehicle_type=self._generate_vehicle_type(self.patterns['morning_rush']),
                        position=self._generate_vehicle_position(lane_name, lane_config),
                        lane=f"{lane_name}_lane_{random.randint(1, lane_config.get('count', 1))}",
                        confidence=random.uniform(0.85, 0.98),
                        timestamp=timestamp
                    )
                    
                    detections.append(detection)
        
        logger.info(f"Simulated {event_type} event with {len(detections)} vehicles")
        return detections
    
    def get_traffic_statistics(self) -> Dict[str, any]:
        """Get current traffic statistics."""
        if not self.active_vehicles:
            return {
                'total_vehicles': 0,
                'vehicles_by_type': {},
                'vehicles_by_lane': {},
                'average_confidence': 0.0
            }
        
        vehicles_by_type = {}
        vehicles_by_lane = {}
        total_confidence = 0.0
        
        for vehicle in self.active_vehicles.values():
            # Count by type
            vehicles_by_type[vehicle.vehicle_type] = vehicles_by_type.get(vehicle.vehicle_type, 0) + 1
            
            # Count by lane
            vehicles_by_lane[vehicle.lane] = vehicles_by_lane.get(vehicle.lane, 0) + 1
            
            # Sum confidence
            total_confidence += vehicle.confidence
        
        return {
            'total_vehicles': len(self.active_vehicles),
            'vehicles_by_type': vehicles_by_type,
            'vehicles_by_lane': vehicles_by_lane,
            'average_confidence': total_confidence / len(self.active_vehicles)
        }
    
    def clear_old_vehicles(self, cutoff_time: datetime) -> None:
        """Remove vehicles older than cutoff time."""
        old_vehicles = [
            vid for vid, vehicle in self.active_vehicles.items()
            if vehicle.timestamp < cutoff_time
        ]
        
        for vid in old_vehicles:
            del self.active_vehicles[vid]
        
        if old_vehicles:
            logger.debug(f"Cleared {len(old_vehicles)} old vehicles")