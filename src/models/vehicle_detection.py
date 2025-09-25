"""VehicleDetection data model with validation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

VALID_VEHICLE_TYPES = {'car', 'truck', 'bus', 'motorcycle'}


@dataclass
class VehicleDetection:
    """Represents a detected vehicle from camera feed processing."""
    
    vehicle_id: str
    vehicle_type: str  # car, truck, bus, motorcycle
    position: Tuple[float, float]  # (x, y) coordinates
    lane: str
    confidence: float
    timestamp: datetime
    
    def __post_init__(self):
        """Validate the vehicle detection data."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all fields in the vehicle detection."""
        if not self.vehicle_id or not isinstance(self.vehicle_id, str):
            raise ValueError("vehicle_id must be a non-empty string")
        
        if not isinstance(self.vehicle_type, str) or self.vehicle_type.lower() not in VALID_VEHICLE_TYPES:
            raise ValueError(f"vehicle_type must be one of {VALID_VEHICLE_TYPES}")
        
        # Normalize vehicle type to lowercase
        self.vehicle_type = self.vehicle_type.lower()
        
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise ValueError("position must be a tuple of two numbers (x, y)")
        
        x, y = self.position
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("position coordinates must be numbers")
        
        # Position coordinates can be negative (west/south directions)
        # Just ensure they are reasonable values (within -1000 to 1000 meters)
        if not (-1000 <= x <= 1000) or not (-1000 <= y <= 1000):
            raise ValueError("position coordinates must be within reasonable range (-1000 to 1000)")
        
        if not self.lane or not isinstance(self.lane, str):
            raise ValueError("lane must be a non-empty string")
        
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be a number between 0.0 and 1.0")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        logger.debug(f"VehicleDetection validated for vehicle {self.vehicle_id}")
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if the detection confidence is above the threshold."""
        return self.confidence >= threshold
    
    def get_distance_from_origin(self) -> float:
        """Calculate the distance from the origin (0, 0)."""
        x, y = self.position
        return (x ** 2 + y ** 2) ** 0.5