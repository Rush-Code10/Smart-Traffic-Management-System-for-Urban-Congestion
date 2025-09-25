"""SignalAction data model with validation."""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalAction:
    """Represents an action to be taken on traffic signals."""
    
    intersection_id: str
    phase_adjustments: Dict[str, int]  # phase -> timing adjustment in seconds
    priority_direction: Optional[str] = None
    reasoning: str = ""
    
    def __post_init__(self):
        """Validate the signal action data."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all fields in the signal action."""
        if not self.intersection_id or not isinstance(self.intersection_id, str):
            raise ValueError("intersection_id must be a non-empty string")
        
        if not isinstance(self.phase_adjustments, dict):
            raise ValueError("phase_adjustments must be a dictionary")
        
        if not self.phase_adjustments:
            raise ValueError("phase_adjustments cannot be empty")
        
        for phase, adjustment in self.phase_adjustments.items():
            if not isinstance(phase, str) or not phase:
                raise ValueError("phase_adjustments keys must be non-empty strings")
            if not isinstance(adjustment, int):
                raise ValueError(f"adjustment for phase {phase} must be an integer")
            if abs(adjustment) > 120:  # Maximum 2 minutes adjustment
                raise ValueError(f"adjustment for phase {phase} must be between -120 and 120 seconds")
        
        if self.priority_direction is not None:
            if not isinstance(self.priority_direction, str) or not self.priority_direction:
                raise ValueError("priority_direction must be None or a non-empty string")
        
        if not isinstance(self.reasoning, str):
            raise ValueError("reasoning must be a string")
        
        logger.debug(f"SignalAction validated for intersection {self.intersection_id}")
    
    def get_total_adjustment(self) -> int:
        """Get the total timing adjustment across all phases."""
        return sum(self.phase_adjustments.values())
    
    def has_priority_direction(self) -> bool:
        """Check if a priority direction is set."""
        return self.priority_direction is not None
    
    def get_max_adjustment(self) -> int:
        """Get the maximum absolute adjustment value."""
        if not self.phase_adjustments:
            return 0
        return max(abs(adj) for adj in self.phase_adjustments.values())