"""Traffic processing modules for vehicle detection, counting, and data aggregation."""

from .traffic_simulator import TrafficSimulator, TrafficPattern
from .camera_processor import CameraProcessor, DetectionBox
from .vehicle_counter import VehicleCounter, VehicleTracker
from .traffic_aggregator import TrafficAggregator, QueueAnalyzer, WaitTimeCalculator

__all__ = [
    'TrafficSimulator',
    'TrafficPattern',
    'CameraProcessor', 
    'DetectionBox',
    'VehicleCounter',
    'VehicleTracker',
    'TrafficAggregator',
    'QueueAnalyzer',
    'WaitTimeCalculator'
]