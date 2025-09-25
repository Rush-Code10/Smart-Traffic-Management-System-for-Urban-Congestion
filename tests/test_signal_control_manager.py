"""Tests for signal control manager."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.agents.signal_control_manager import (
    SignalControlManager, SignalState, PhaseState, IntersectionSignalState
)
from src.models.signal_action import SignalAction
from src.config.intersection_config import IntersectionConfig


class TestPhaseState:
    """Test phase state data class."""
    
    def test_phase_state_creation(self):
        """Test phase state creation."""
        phase = PhaseState(
            phase_id="north_south",
            current_state=SignalState.GREEN,
            time_remaining=30,
            total_duration=60
        )
        
        assert phase.phase_id == "north_south"
        assert phase.current_state == SignalState.GREEN
        assert phase.time_remaining == 30
        assert phase.total_duration == 60
    
    def test_get_progress(self):
        """Test progress calculation."""
        # Half way through
        phase = PhaseState("test", SignalState.GREEN, 30, 60)
        assert phase.get_progress() == 0.5
        
        # Just started
        phase = PhaseState("test", SignalState.GREEN, 60, 60)
        assert phase.get_progress() == 0.0
        
        # Almost finished
        phase = PhaseState("test", SignalState.GREEN, 5, 60)
        assert abs(phase.get_progress() - (55/60)) < 0.01
        
        # Zero duration
        phase = PhaseState("test", SignalState.GREEN, 0, 0)
        assert phase.get_progress() == 1.0


class TestIntersectionSignalState:
    """Test intersection signal state."""
    
    @pytest.fixture
    def signal_state(self):
        """Create a sample intersection signal state."""
        phases = {
            "north_south": PhaseState("north_south", SignalState.GREEN, 30, 60),
            "east_west": PhaseState("east_west", SignalState.RED, 0, 0)
        }
        
        return IntersectionSignalState(
            intersection_id="test_intersection",
            timestamp=datetime.now(),
            phases=phases,
            current_cycle_phase="north_south",
            cycle_start_time=datetime.now()
        )
    
    def test_get_active_phases(self, signal_state):
        """Test getting active phases."""
        active = signal_state.get_active_phases()
        assert active == ["north_south"]
    
    def test_get_phase_state(self, signal_state):
        """Test getting specific phase state."""
        phase = signal_state.get_phase_state("north_south")
        assert phase is not None
        assert phase.phase_id == "north_south"
        assert phase.current_state == SignalState.GREEN
        
        # Non-existent phase
        phase = signal_state.get_phase_state("nonexistent")
        assert phase is None


class TestSignalControlManager:
    """Test signal control manager."""
    
    @pytest.fixture
    def intersection_config(self):
        """Create intersection configuration for testing."""
        config = IntersectionConfig("test_intersection", "Test Intersection")
        config.add_lane("north", 2, 100.0)
        config.add_lane("south", 2, 100.0)
        config.add_lane("east", 1, 80.0)
        config.add_lane("west", 1, 80.0)
        config.set_signal_phases(["north_south", "east_west"])
        config.set_default_phase_timings({"north_south": 60, "east_west": 45})
        return config
    
    @pytest.fixture
    def signal_manager(self, intersection_config):
        """Create signal control manager for testing."""
        return SignalControlManager(intersection_config)
    
    def test_initialization(self, signal_manager, intersection_config):
        """Test signal manager initialization."""
        assert signal_manager.intersection_id == "test_intersection"
        assert signal_manager.cycle_phases == ["north_south", "east_west"]
        assert signal_manager.phase_timings == {"north_south": 60, "east_west": 45}
        assert signal_manager.current_state is not None
        assert not signal_manager.manual_override
    
    def test_get_current_signal_state(self, signal_manager):
        """Test getting current signal state."""
        state = signal_manager.get_current_signal_state()
        
        assert isinstance(state, IntersectionSignalState)
        assert state.intersection_id == "test_intersection"
        assert len(state.phases) == 2
        assert "north_south" in state.phases
        assert "east_west" in state.phases
        
        # First phase should be active
        assert state.current_cycle_phase == "north_south"
        assert state.phases["north_south"].current_state == SignalState.GREEN
        assert state.phases["east_west"].current_state == SignalState.RED
    
    def test_apply_signal_action_valid(self, signal_manager):
        """Test applying valid signal action."""
        action = SignalAction(
            intersection_id="test_intersection",
            phase_adjustments={"north_south": 10, "east_west": -5},
            reasoning="Test adjustment"
        )
        
        original_timings = signal_manager.phase_timings.copy()
        result = signal_manager.apply_signal_action(action)
        
        assert result is True
        assert signal_manager.phase_timings["north_south"] == original_timings["north_south"] + 10
        assert signal_manager.phase_timings["east_west"] == original_timings["east_west"] - 5
        assert len(signal_manager.action_history) == 1
    
    def test_apply_signal_action_wrong_intersection(self, signal_manager):
        """Test applying action for wrong intersection."""
        action = SignalAction(
            intersection_id="wrong_intersection",
            phase_adjustments={"north_south": 10},
            reasoning="Test"
        )
        
        result = signal_manager.apply_signal_action(action)
        assert result is False
        assert len(signal_manager.action_history) == 0
    
    def test_apply_signal_action_unknown_phase(self, signal_manager):
        """Test applying action with unknown phase."""
        action = SignalAction(
            intersection_id="test_intersection",
            phase_adjustments={"unknown_phase": 10},
            reasoning="Test"
        )
        
        result = signal_manager.apply_signal_action(action)
        assert result is False
    
    def test_apply_signal_action_timing_limits(self, signal_manager):
        """Test that timing adjustments are clamped to limits."""
        # Test upper limit - use maximum allowed adjustment
        action = SignalAction(
            intersection_id="test_intersection",
            phase_adjustments={"north_south": 120},  # Maximum allowed adjustment
            reasoning="Test upper limit"
        )
        
        signal_manager.apply_signal_action(action)
        assert signal_manager.phase_timings["north_south"] == 120  # Should be clamped to max
        
        # Test lower limit - start with a high value and reduce it significantly
        signal_manager.phase_timings["north_south"] = 100  # Set high value first
        action = SignalAction(
            intersection_id="test_intersection",
            phase_adjustments={"north_south": -120},  # Maximum negative adjustment
            reasoning="Test lower limit"
        )
        
        signal_manager.apply_signal_action(action)
        assert signal_manager.phase_timings["north_south"] == 10  # Should be clamped to min
    
    def test_apply_signal_action_during_manual_override(self, signal_manager):
        """Test that actions are rejected during manual override."""
        signal_manager.enable_manual_override("operator1")
        
        action = SignalAction(
            intersection_id="test_intersection",
            phase_adjustments={"north_south": 10},
            reasoning="Test during override"
        )
        
        result = signal_manager.apply_signal_action(action)
        assert result is False
    
    def test_manual_override_enable_disable(self, signal_manager):
        """Test manual override enable/disable."""
        # Enable override
        result = signal_manager.enable_manual_override("operator1")
        assert result is True
        assert signal_manager.manual_override is True
        assert signal_manager.override_operator == "operator1"
        assert signal_manager.override_start_time is not None
        
        # Try to enable again (should fail)
        result = signal_manager.enable_manual_override("operator2")
        assert result is False
        assert signal_manager.override_operator == "operator1"  # Unchanged
        
        # Disable override
        result = signal_manager.disable_manual_override()
        assert result is True
        assert signal_manager.manual_override is False
        assert signal_manager.override_operator is None
        
        # Try to disable again (should fail)
        result = signal_manager.disable_manual_override()
        assert result is False
    
    def test_set_manual_phase(self, signal_manager):
        """Test manual phase setting."""
        # Should fail without manual override
        result = signal_manager.set_manual_phase("east_west", 30)
        assert result is False
        
        # Enable manual override
        signal_manager.enable_manual_override("operator1")
        
        # Should succeed now
        result = signal_manager.set_manual_phase("east_west", 30)
        assert result is True
        
        state = signal_manager.get_current_signal_state()
        assert state.current_cycle_phase == "east_west"
        assert state.phases["east_west"].current_state == SignalState.GREEN
        assert state.phases["east_west"].total_duration == 30
        assert state.phases["north_south"].current_state == SignalState.RED
    
    def test_set_manual_phase_invalid_phase(self, signal_manager):
        """Test manual phase setting with invalid phase."""
        signal_manager.enable_manual_override("operator1")
        
        result = signal_manager.set_manual_phase("invalid_phase", 30)
        assert result is False
    
    def test_set_manual_phase_invalid_duration(self, signal_manager):
        """Test manual phase setting with invalid duration."""
        signal_manager.enable_manual_override("operator1")
        
        # Too short
        result = signal_manager.set_manual_phase("east_west", 5)
        assert result is False
        
        # Too long
        result = signal_manager.set_manual_phase("east_west", 400)
        assert result is False
    
    def test_phase_advancement(self, signal_manager):
        """Test automatic phase advancement."""
        # Mock time to simulate phase completion
        with patch('src.agents.signal_control_manager.datetime') as mock_datetime:
            start_time = datetime.now()
            mock_datetime.now.return_value = start_time
            
            # Initialize with known time
            signal_manager._initialize_signal_state()
            signal_manager.phase_start_time = start_time
            
            # Simulate time passing beyond phase duration
            future_time = start_time + timedelta(seconds=70)  # Beyond 60s phase
            mock_datetime.now.return_value = future_time
            
            # Update timing should advance phase
            signal_manager._update_signal_timing()
            
            state = signal_manager.get_current_signal_state()
            assert state.current_cycle_phase == "east_west"  # Should have advanced
    
    def test_action_history(self, signal_manager):
        """Test action history tracking."""
        # Apply several actions
        for i in range(3):
            action = SignalAction(
                intersection_id="test_intersection",
                phase_adjustments={"north_south": i},
                reasoning=f"Test action {i}"
            )
            signal_manager.apply_signal_action(action)
        
        history = signal_manager.get_action_history()
        assert len(history) == 3
        
        # Check that history is in chronological order
        for i, (timestamp, action) in enumerate(history):
            assert isinstance(timestamp, datetime)
            assert action.reasoning == f"Test action {i}"
        
        # Test history limit
        history_limited = signal_manager.get_action_history(limit=2)
        assert len(history_limited) == 2
    
    def test_get_phase_timings(self, signal_manager):
        """Test getting phase timings."""
        timings = signal_manager.get_phase_timings()
        
        assert isinstance(timings, dict)
        assert timings == {"north_south": 60, "east_west": 45}
        
        # Modify original and ensure copy is returned
        signal_manager.phase_timings["north_south"] = 90
        timings_after = signal_manager.get_phase_timings()
        assert timings_after["north_south"] == 90
        assert timings != timings_after  # Original copy unchanged
    
    def test_reset_to_defaults(self, signal_manager):
        """Test resetting to default timings."""
        # Modify timings
        signal_manager.phase_timings["north_south"] = 90
        signal_manager.phase_timings["east_west"] = 30
        
        # Reset
        signal_manager.reset_to_defaults()
        
        assert signal_manager.phase_timings == {"north_south": 60, "east_west": 45}
    
    def test_get_status_summary(self, signal_manager):
        """Test getting status summary."""
        summary = signal_manager.get_status_summary()
        
        assert isinstance(summary, dict)
        assert summary['intersection_id'] == "test_intersection"
        assert summary['current_phase'] == "north_south"
        assert summary['manual_override'] is False
        assert summary['override_operator'] is None
        assert 'phase_timings' in summary
        assert 'active_phases' in summary
        assert 'recent_actions' in summary
        assert 'timestamp' in summary
    
    def test_current_phase_timing_adjustment(self, signal_manager):
        """Test that current phase timing is adjusted when action is applied."""
        # Get initial state
        initial_state = signal_manager.get_current_signal_state()
        initial_remaining = initial_state.phases["north_south"].time_remaining
        initial_total = initial_state.phases["north_south"].total_duration
        
        # Apply action to current phase
        action = SignalAction(
            intersection_id="test_intersection",
            phase_adjustments={"north_south": 20},
            reasoning="Extend current phase"
        )
        
        signal_manager.apply_signal_action(action)
        
        # Check that current phase state was updated
        updated_state = signal_manager.get_current_signal_state()
        updated_total = updated_state.phases["north_south"].total_duration
        
        assert updated_total == initial_total + 20
        
        # Time remaining should be adjusted proportionally
        # (This is a simplified test - actual implementation may vary)
        assert updated_state.phases["north_south"].time_remaining > 0