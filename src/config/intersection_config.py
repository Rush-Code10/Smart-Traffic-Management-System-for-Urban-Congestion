"""Intersection-specific configuration management."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class LaneConfig:
    """Configuration for a single lane."""
    count: int  # Number of lanes in this direction
    length: float  # Length of the lane in meters
    
    def validate(self) -> None:
        """Validate lane configuration."""
        if not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("Lane count must be a positive integer")
        if not isinstance(self.length, (int, float)) or self.length <= 0:
            raise ValueError("Lane length must be a positive number")


@dataclass
class CameraPosition:
    """Configuration for camera position and orientation."""
    x: float  # X coordinate
    y: float  # Y coordinate
    angle: float  # Viewing angle in degrees
    
    def validate(self) -> None:
        """Validate camera position configuration."""
        if not isinstance(self.x, (int, float)):
            raise ValueError("Camera x coordinate must be a number")
        if not isinstance(self.y, (int, float)):
            raise ValueError("Camera y coordinate must be a number")
        if not isinstance(self.angle, (int, float)) or not (0 <= self.angle < 360):
            raise ValueError("Camera angle must be between 0 and 360 degrees")


class IntersectionConfig:
    """Configuration for a traffic intersection."""
    
    def __init__(self, intersection_id: str, name: str):
        self.intersection_id = intersection_id
        self.name = name
        self.lanes: Dict[str, LaneConfig] = {}
        self.signal_phases: List[str] = []
        self.default_phase_timings: Dict[str, int] = {}
        self.camera_positions: Dict[str, CameraPosition] = {}
    
    def add_lane(self, direction: str, count: int, length: float) -> None:
        """Add lane configuration for a direction."""
        lane_config = LaneConfig(count=count, length=length)
        lane_config.validate()
        self.lanes[direction] = lane_config
        logger.info(f"Added lane config for {self.intersection_id} - {direction}: {count} lanes, {length}m")
    
    def add_camera_position(self, direction: str, x: float, y: float, angle: float) -> None:
        """Add camera position for a direction."""
        camera_pos = CameraPosition(x=x, y=y, angle=angle)
        camera_pos.validate()
        self.camera_positions[direction] = camera_pos
        logger.info(f"Added camera position for {self.intersection_id} - {direction}: ({x}, {y}) at {angle}°")
    
    def set_signal_phases(self, phases: List[str]) -> None:
        """Set the signal phases for this intersection."""
        if not isinstance(phases, list) or not phases:
            raise ValueError("Signal phases must be a non-empty list")
        
        for phase in phases:
            if not isinstance(phase, str) or not phase:
                raise ValueError("Each signal phase must be a non-empty string")
        
        self.signal_phases = phases
        logger.info(f"Set signal phases for {self.intersection_id}: {phases}")
    
    def set_default_phase_timings(self, timings: Dict[str, int]) -> None:
        """Set default timing for each signal phase."""
        if not isinstance(timings, dict) or not timings:
            raise ValueError("Phase timings must be a non-empty dictionary")
        
        for phase, timing in timings.items():
            if not isinstance(phase, str) or not phase:
                raise ValueError("Phase names must be non-empty strings")
            if not isinstance(timing, int) or timing <= 0:
                raise ValueError(f"Timing for phase {phase} must be a positive integer")
        
        self.default_phase_timings = timings
        logger.info(f"Set default phase timings for {self.intersection_id}: {timings}")
    
    def get_total_lanes(self) -> int:
        """Get total number of lanes across all directions."""
        return sum(lane.count for lane in self.lanes.values())
    
    def get_directions(self) -> List[str]:
        """Get list of all configured directions."""
        return list(self.lanes.keys())
    
    def get_lane_count(self, direction: str) -> int:
        """Get number of lanes for a specific direction."""
        if direction not in self.lanes:
            raise ValueError(f"Direction {direction} not configured")
        return self.lanes[direction].count
    
    def get_lane_length(self, direction: str) -> float:
        """Get lane length for a specific direction."""
        if direction not in self.lanes:
            raise ValueError(f"Direction {direction} not configured")
        return self.lanes[direction].length
    
    def has_camera(self, direction: str) -> bool:
        """Check if a camera is configured for a direction."""
        return direction in self.camera_positions
    
    def get_camera_position(self, direction: str) -> Tuple[float, float, float]:
        """Get camera position and angle for a direction."""
        if direction not in self.camera_positions:
            raise ValueError(f"Camera not configured for direction {direction}")
        
        camera = self.camera_positions[direction]
        return (camera.x, camera.y, camera.angle)
    
    def validate(self) -> None:
        """Validate the complete intersection configuration."""
        if not self.intersection_id or not isinstance(self.intersection_id, str):
            raise ValueError("intersection_id must be a non-empty string")
        
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string")
        
        if not self.lanes:
            raise ValueError("At least one lane direction must be configured")
        
        if not self.signal_phases:
            raise ValueError("At least one signal phase must be configured")
        
        if not self.default_phase_timings:
            raise ValueError("Default phase timings must be configured")
        
        # Validate that all lanes have corresponding cameras
        for direction in self.lanes.keys():
            if direction not in self.camera_positions:
                logger.warning(f"No camera configured for direction {direction}")
        
        logger.info(f"Intersection configuration validated for {self.intersection_id}")
    
    def to_dict(self) -> Dict:
        """Convert intersection config to dictionary format."""
        return {
            'name': self.name,
            'geometry': {
                'lanes': {
                    direction: {'count': lane.count, 'length': lane.length}
                    for direction, lane in self.lanes.items()
                },
                'signal_phases': self.signal_phases,
                'default_phase_timings': self.default_phase_timings
            },
            'camera_positions': {
                direction: {'x': cam.x, 'y': cam.y, 'angle': cam.angle}
                for direction, cam in self.camera_positions.items()
            }
        }