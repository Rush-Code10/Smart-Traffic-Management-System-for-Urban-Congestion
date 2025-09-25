"""TrafficState data model with validation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrafficState:
    """Represents the current traffic state at an intersection."""
    
    intersection_id: str
    timestamp: datetime
    vehicle_counts: Dict[str, int]  # direction -> count
    queue_lengths: Dict[str, float]  # direction -> length in meters
    wait_times: Dict[str, float]    # direction -> average wait in seconds
    signal_phase: str               # current signal phase
    prediction_confidence: float = field(default=0.0)
    
    def __post_init__(self):
        """Validate the traffic state data."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all fields in the traffic state."""
        if not self.intersection_id or not isinstance(self.intersection_id, str):
            raise ValueError("intersection_id must be a non-empty string")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        if not isinstance(self.vehicle_counts, dict):
            raise ValueError("vehicle_counts must be a dictionary")
        
        for direction, count in self.vehicle_counts.items():
            if not isinstance(direction, str) or not direction:
                raise ValueError("vehicle_counts keys must be non-empty strings")
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"vehicle count for {direction} must be a non-negative integer")
        
        if not isinstance(self.queue_lengths, dict):
            raise ValueError("queue_lengths must be a dictionary")
        
        for direction, length in self.queue_lengths.items():
            if not isinstance(direction, str) or not direction:
                raise ValueError("queue_lengths keys must be non-empty strings")
            if not isinstance(length, (int, float)) or length < 0:
                raise ValueError(f"queue length for {direction} must be a non-negative number")
        
        if not isinstance(self.wait_times, dict):
            raise ValueError("wait_times must be a dictionary")
        
        for direction, wait_time in self.wait_times.items():
            if not isinstance(direction, str) or not direction:
                raise ValueError("wait_times keys must be non-empty strings")
            if not isinstance(wait_time, (int, float)) or wait_time < 0:
                raise ValueError(f"wait time for {direction} must be a non-negative number")
        
        if not isinstance(self.signal_phase, str) or not self.signal_phase:
            raise ValueError("signal_phase must be a non-empty string")
        
        if not isinstance(self.prediction_confidence, (int, float)) or not (0.0 <= self.prediction_confidence <= 1.0):
            raise ValueError("prediction_confidence must be a number between 0.0 and 1.0")
        
        logger.debug(f"TrafficState validated for intersection {self.intersection_id}")
    
    def get_total_vehicles(self) -> int:
        """Get the total number of vehicles across all directions."""
        return sum(self.vehicle_counts.values())
    
    def get_total_queue_length(self) -> float:
        """Get the total queue length across all directions."""
        return sum(self.queue_lengths.values())
    
    def get_average_wait_time(self) -> float:
        """Get the average wait time across all directions."""
        if not self.wait_times:
            return 0.0
        return sum(self.wait_times.values()) / len(self.wait_times)