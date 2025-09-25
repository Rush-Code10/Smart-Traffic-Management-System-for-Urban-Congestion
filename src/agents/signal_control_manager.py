"""Signal control manager for interfacing with traffic signals."""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..models.signal_action import SignalAction
from ..config.intersection_config import IntersectionConfig

logger = logging.getLogger(__name__)


class SignalState(Enum):
    """Traffic signal states."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass
class PhaseState:
    """State of a signal phase."""
    phase_id: str
    current_state: SignalState
    time_remaining: int  # seconds remaining in current state
    total_duration: int  # total duration of current state
    
    def get_progress(self) -> float:
        """Get progress through current state (0.0 to 1.0)."""
        if self.total_duration <= 0:
            return 1.0
        return max(0.0, 1.0 - (self.time_remaining / self.total_duration))


@dataclass
class IntersectionSignalState:
    """Complete signal state for an intersection."""
    intersection_id: str
    timestamp: datetime
    phases: Dict[str, PhaseState]
    current_cycle_phase: str
    cycle_start_time: datetime
    manual_override: bool = False
    override_operator: Optional[str] = None
    
    def get_active_phases(self) -> List[str]:
        """Get list of phases currently showing green."""
        return [phase_id for phase_id, phase in self.phases.items() 
                if phase.current_state == SignalState.GREEN]
    
    def get_phase_state(self, phase_id: str) -> Optional[PhaseState]:
        """Get state of a specific phase."""
        return self.phases.get(phase_id)


class SignalControlManager:
    """Manages traffic signal control for intersections."""
    
    def __init__(self, intersection_config: IntersectionConfig):
        """Initialize signal control manager.
        
        Args:
            intersection_config: Configuration for the intersection
        """
        self.intersection_config = intersection_config
        self.intersection_id = intersection_config.intersection_id
        
        # Current signal state
        self.current_state: Optional[IntersectionSignalState] = None
        
        # Signal timing configuration
        self.phase_timings = intersection_config.default_phase_timings.copy()
        self.yellow_duration = 3  # seconds
        self.all_red_duration = 2  # seconds
        
        # Cycle management
        self.cycle_phases = intersection_config.signal_phases
        self.current_phase_index = 0
        self.phase_start_time = datetime.now()
        
        # Manual override state
        self.manual_override = False
        self.override_operator: Optional[str] = None
        self.override_start_time: Optional[datetime] = None
        
        # Action history for logging
        self.action_history: List[Tuple[datetime, SignalAction]] = []
        
        # Initialize signal state
        self._initialize_signal_state()
        
        logger.info(f"Signal control manager initialized for intersection {self.intersection_id}")
        logger.info(f"Phases: {self.cycle_phases}")
        logger.info(f"Default timings: {self.phase_timings}")
    
    def _initialize_signal_state(self) -> None:
        """Initialize the signal state with default values."""
        now = datetime.now()
        phases = {}
        
        for i, phase_id in enumerate(self.cycle_phases):
            if i == self.current_phase_index:
                # Current phase is green
                state = SignalState.GREEN
                duration = self.phase_timings.get(phase_id, 30)
                remaining = duration
            else:
                # Other phases are red
                state = SignalState.RED
                duration = 0
                remaining = 0
            
            phases[phase_id] = PhaseState(
                phase_id=phase_id,
                current_state=state,
                time_remaining=remaining,
                total_duration=duration
            )
        
        self.current_state = IntersectionSignalState(
            intersection_id=self.intersection_id,
            timestamp=now,
            phases=phases,
            current_cycle_phase=self.cycle_phases[self.current_phase_index],
            cycle_start_time=now
        )
    
    def get_current_signal_state(self) -> IntersectionSignalState:
        """Get the current signal state."""
        if self.current_state is None:
            self._initialize_signal_state()
        
        # Update timing information
        self._update_signal_timing()
        return self.current_state
    
    def _update_signal_timing(self) -> None:
        """Update signal timing based on elapsed time."""
        if self.current_state is None:
            return
        
        now = datetime.now()
        elapsed = (now - self.phase_start_time).total_seconds()
        
        current_phase = self.cycle_phases[self.current_phase_index]
        current_phase_state = self.current_state.phases[current_phase]
        
        # Update time remaining
        time_remaining = max(0, current_phase_state.total_duration - int(elapsed))
        current_phase_state.time_remaining = time_remaining
        
        # Check if phase should transition
        if time_remaining <= 0 and not self.manual_override:
            self._advance_to_next_phase()
        
        # Update timestamp
        self.current_state.timestamp = now
    
    def _advance_to_next_phase(self) -> None:
        """Advance to the next phase in the cycle."""
        if self.current_state is None:
            return
        
        # Move to next phase
        self.current_phase_index = (self.current_phase_index + 1) % len(self.cycle_phases)
        next_phase = self.cycle_phases[self.current_phase_index]
        
        # Update phase states
        for phase_id, phase_state in self.current_state.phases.items():
            if phase_id == next_phase:
                # New active phase
                phase_state.current_state = SignalState.GREEN
                phase_state.total_duration = self.phase_timings.get(phase_id, 30)
                phase_state.time_remaining = phase_state.total_duration
            else:
                # Inactive phases
                phase_state.current_state = SignalState.RED
                phase_state.total_duration = 0
                phase_state.time_remaining = 0
        
        # Update cycle information
        self.current_state.current_cycle_phase = next_phase
        self.phase_start_time = datetime.now()
        
        logger.info(f"Advanced to phase {next_phase} with duration {self.phase_timings.get(next_phase, 30)}s")
    
    def apply_signal_action(self, action: SignalAction) -> bool:
        """Apply a signal timing action.
        
        Args:
            action: Signal action to apply
            
        Returns:
            True if action was applied successfully, False otherwise
        """
        if action.intersection_id != self.intersection_id:
            logger.error(f"Action intersection ID {action.intersection_id} does not match {self.intersection_id}")
            return False
        
        if self.manual_override:
            logger.warning(f"Cannot apply action - intersection {self.intersection_id} is in manual override mode")
            return False
        
        # Validate phase adjustments
        for phase_id, adjustment in action.phase_adjustments.items():
            if phase_id not in self.cycle_phases:
                logger.error(f"Unknown phase {phase_id} in action")
                return False
        
        # Apply timing adjustments
        adjustments_applied = []
        for phase_id, adjustment in action.phase_adjustments.items():
            if adjustment != 0:
                old_timing = self.phase_timings.get(phase_id, 30)
                new_timing = max(10, min(120, old_timing + adjustment))  # Clamp between 10-120 seconds
                self.phase_timings[phase_id] = new_timing
                adjustments_applied.append(f"{phase_id}: {old_timing}s -> {new_timing}s")
                
                # If this is the current phase, update the current state
                if (self.current_state and 
                    phase_id == self.current_state.current_cycle_phase):
                    current_phase_state = self.current_state.phases[phase_id]
                    # Adjust remaining time proportionally
                    progress = current_phase_state.get_progress()
                    current_phase_state.total_duration = new_timing
                    current_phase_state.time_remaining = int(new_timing * (1 - progress))
        
        # Log the action
        self.action_history.append((datetime.now(), action))
        
        # Keep only last 100 actions for memory efficiency
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        if adjustments_applied:
            logger.info(f"Applied signal action: {', '.join(adjustments_applied)}")
            logger.info(f"Reasoning: {action.reasoning}")
        else:
            logger.info("No timing adjustments applied (all adjustments were 0)")
        
        return True
    
    def enable_manual_override(self, operator_id: str) -> bool:
        """Enable manual override mode.
        
        Args:
            operator_id: ID of the operator taking control
            
        Returns:
            True if override was enabled successfully
        """
        if self.manual_override:
            logger.warning(f"Manual override already enabled by {self.override_operator}")
            return False
        
        self.manual_override = True
        self.override_operator = operator_id
        self.override_start_time = datetime.now()
        
        if self.current_state:
            self.current_state.manual_override = True
            self.current_state.override_operator = operator_id
        
        logger.info(f"Manual override enabled by operator {operator_id}")
        return True
    
    def disable_manual_override(self) -> bool:
        """Disable manual override mode and resume automatic control.
        
        Returns:
            True if override was disabled successfully
        """
        if not self.manual_override:
            logger.warning("Manual override is not currently enabled")
            return False
        
        override_duration = None
        if self.override_start_time:
            override_duration = (datetime.now() - self.override_start_time).total_seconds()
        
        self.manual_override = False
        operator = self.override_operator
        self.override_operator = None
        self.override_start_time = None
        
        if self.current_state:
            self.current_state.manual_override = False
            self.current_state.override_operator = None
        
        # Reset phase timing to resume normal cycle
        self.phase_start_time = datetime.now()
        
        logger.info(f"Manual override disabled. Operator {operator} had control for {override_duration:.1f}s")
        return True
    
    def set_manual_phase(self, phase_id: str, duration: int) -> bool:
        """Manually set a specific phase (only works in manual override mode).
        
        Args:
            phase_id: ID of the phase to activate
            duration: Duration in seconds
            
        Returns:
            True if phase was set successfully
        """
        if not self.manual_override:
            logger.error("Manual phase control requires manual override mode")
            return False
        
        if phase_id not in self.cycle_phases:
            logger.error(f"Unknown phase {phase_id}")
            return False
        
        if not (10 <= duration <= 300):
            logger.error(f"Duration {duration} must be between 10 and 300 seconds")
            return False
        
        if self.current_state is None:
            return False
        
        # Update all phases
        for pid, phase_state in self.current_state.phases.items():
            if pid == phase_id:
                phase_state.current_state = SignalState.GREEN
                phase_state.total_duration = duration
                phase_state.time_remaining = duration
            else:
                phase_state.current_state = SignalState.RED
                phase_state.total_duration = 0
                phase_state.time_remaining = 0
        
        self.current_state.current_cycle_phase = phase_id
        self.phase_start_time = datetime.now()
        
        logger.info(f"Manual phase set: {phase_id} for {duration}s by operator {self.override_operator}")
        return True
    
    def get_action_history(self, limit: int = 10) -> List[Tuple[datetime, SignalAction]]:
        """Get recent action history.
        
        Args:
            limit: Maximum number of actions to return
            
        Returns:
            List of (timestamp, action) tuples
        """
        return self.action_history[-limit:] if self.action_history else []
    
    def get_phase_timings(self) -> Dict[str, int]:
        """Get current phase timings."""
        return self.phase_timings.copy()
    
    def reset_to_defaults(self) -> None:
        """Reset phase timings to default values."""
        self.phase_timings = self.intersection_config.default_phase_timings.copy()
        logger.info(f"Phase timings reset to defaults: {self.phase_timings}")
    
    def get_status_summary(self) -> Dict:
        """Get a summary of the current signal control status."""
        state = self.get_current_signal_state()
        
        return {
            'intersection_id': self.intersection_id,
            'current_phase': state.current_cycle_phase,
            'manual_override': self.manual_override,
            'override_operator': self.override_operator,
            'phase_timings': self.phase_timings,
            'active_phases': state.get_active_phases(),
            'recent_actions': len(self.action_history),
            'timestamp': state.timestamp.isoformat()
        }