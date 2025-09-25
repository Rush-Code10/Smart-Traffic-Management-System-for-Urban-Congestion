"""Tests for Q-learning agent."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from src.agents.q_learning_agent import QLearningAgent, QLearningConfig
from src.models.traffic_state import TrafficState
from src.models.signal_action import SignalAction


class TestQLearningConfig:
    """Test Q-learning configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QLearningConfig()
        assert config.learning_rate == 0.1
        assert config.discount_factor == 0.95
        assert config.epsilon == 0.1
        assert config.epsilon_decay == 0.995
        assert config.epsilon_min == 0.01
        assert config.max_adjustment == 30
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = QLearningConfig(learning_rate=0.5, discount_factor=0.9)
        config.validate()
        
        # Invalid learning rate
        with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
            QLearningConfig(learning_rate=0.0).validate()
        
        with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
            QLearningConfig(learning_rate=1.5).validate()
        
        # Invalid discount factor
        with pytest.raises(ValueError, match="discount_factor must be between 0 and 1"):
            QLearningConfig(discount_factor=-0.1).validate()
        
        with pytest.raises(ValueError, match="discount_factor must be between 0 and 1"):
            QLearningConfig(discount_factor=1.1).validate()
        
        # Invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            QLearningConfig(epsilon=-0.1).validate()
        
        # Invalid max_adjustment
        with pytest.raises(ValueError, match="max_adjustment must be a positive integer"):
            QLearningConfig(max_adjustment=0).validate()


class TestQLearningAgent:
    """Test Q-learning agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a Q-learning agent for testing."""
        return QLearningAgent(
            intersection_id="test_intersection",
            signal_phases=["north_south", "east_west"],
            config=QLearningConfig(epsilon=0.1, max_adjustment=20)
        )
    
    @pytest.fixture
    def traffic_state(self):
        """Create a sample traffic state."""
        return TrafficState(
            intersection_id="test_intersection",
            timestamp=datetime.now(),
            vehicle_counts={"north": 10, "south": 8, "east": 5, "west": 7},
            queue_lengths={"north": 25.0, "south": 20.0, "east": 15.0, "west": 18.0},
            wait_times={"north": 45.0, "south": 40.0, "east": 30.0, "west": 35.0},
            signal_phase="north_south",
            prediction_confidence=0.8
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.intersection_id == "test_intersection"
        assert agent.signal_phases == ["north_south", "east_west"]
        assert len(agent.actions) > 0
        assert agent.training_episodes == 0
        assert len(agent.q_table) == 0
    
    def test_action_space_generation(self, agent):
        """Test action space generation."""
        actions = agent.actions
        
        # Should have actions for each phase plus no-change action
        assert len(actions) > 0
        
        # Check that no-change action exists
        no_change_action = {"north_south": 0, "east_west": 0}
        assert no_change_action in actions
        
        # Check that individual phase adjustments exist
        found_north_south_adjustment = False
        found_east_west_adjustment = False
        
        for action in actions:
            if action["north_south"] != 0 and action["east_west"] == 0:
                found_north_south_adjustment = True
            if action["east_west"] != 0 and action["north_south"] == 0:
                found_east_west_adjustment = True
        
        assert found_north_south_adjustment
        assert found_east_west_adjustment
    
    def test_state_discretization(self, agent, traffic_state):
        """Test traffic state discretization."""
        state_str = agent._discretize_state(traffic_state)
        
        # Should be a string with vehicle, queue, wait, and phase components
        assert isinstance(state_str, str)
        assert "v" in state_str  # vehicle count bin
        assert "q" in state_str  # queue length bin
        assert "w" in state_str  # wait time bin
        assert "p" in state_str  # phase
        assert "north_south" in state_str
    
    def test_bin_index_calculation(self, agent):
        """Test bin index calculation."""
        bins = [0, 10, 20, 50]
        
        assert agent._get_bin_index(5, bins) == 0
        assert agent._get_bin_index(15, bins) == 1
        assert agent._get_bin_index(25, bins) == 2
        assert agent._get_bin_index(100, bins) == 3  # Above highest bin
    
    def test_get_action(self, agent, traffic_state):
        """Test action selection."""
        action = agent.get_action(traffic_state)
        
        assert isinstance(action, SignalAction)
        assert action.intersection_id == "test_intersection"
        assert isinstance(action.phase_adjustments, dict)
        assert len(action.phase_adjustments) == 2  # Two phases
        assert "north_south" in action.phase_adjustments
        assert "east_west" in action.phase_adjustments
        assert isinstance(action.reasoning, str)
        assert len(action.reasoning) > 0
    
    def test_action_selection_exploration(self, agent, traffic_state):
        """Test that exploration occurs with high epsilon."""
        agent.config.epsilon = 1.0  # Always explore
        
        actions = []
        for _ in range(10):
            action = agent.get_action(traffic_state)
            actions.append(action.phase_adjustments)
        
        # With high epsilon, we should see some variation in actions
        unique_actions = len(set(str(a) for a in actions))
        assert unique_actions > 1  # Should have some variety
    
    def test_action_selection_exploitation(self, agent, traffic_state):
        """Test that exploitation occurs with low epsilon."""
        agent.config.epsilon = 0.0  # Never explore
        
        # Initialize Q-table with known values
        state_str = agent._discretize_state(traffic_state)
        agent.q_table[state_str] = {i: 0.0 for i in range(len(agent.actions))}
        agent.q_table[state_str][0] = 10.0  # Make first action best
        
        action = agent.get_action(traffic_state)
        
        # Should select the action with highest Q-value
        assert action.phase_adjustments == agent.actions[0]
    
    def test_reward_calculation(self, agent, traffic_state):
        """Test reward calculation."""
        # Create a better traffic state (lower wait times and queue lengths)
        better_state = TrafficState(
            intersection_id="test_intersection",
            timestamp=datetime.now(),
            vehicle_counts={"north": 10, "south": 8, "east": 5, "west": 7},
            queue_lengths={"north": 20.0, "south": 15.0, "east": 10.0, "west": 13.0},
            wait_times={"north": 35.0, "south": 30.0, "east": 20.0, "west": 25.0},
            signal_phase="north_south",
            prediction_confidence=0.8
        )
        
        action = {"north_south": 5, "east_west": 0}
        reward = agent._calculate_reward(traffic_state, better_state, action)
        
        # Should be positive since traffic improved
        assert reward > 0
    
    def test_reward_calculation_penalty(self, agent, traffic_state):
        """Test reward calculation with large adjustment penalty."""
        # Same traffic state (no improvement)
        same_state = TrafficState(
            intersection_id="test_intersection",
            timestamp=datetime.now(),
            vehicle_counts=traffic_state.vehicle_counts,
            queue_lengths=traffic_state.queue_lengths,
            wait_times=traffic_state.wait_times,
            signal_phase=traffic_state.signal_phase,
            prediction_confidence=traffic_state.prediction_confidence
        )
        
        # Large adjustment should result in penalty
        large_action = {"north_south": 30, "east_west": -30}
        reward = agent._calculate_reward(traffic_state, same_state, large_action)
        
        # Should be negative due to large adjustment penalty
        assert reward < 0
    
    def test_q_value_update(self, agent, traffic_state):
        """Test Q-value updates."""
        # Create next state
        next_state = TrafficState(
            intersection_id="test_intersection",
            timestamp=datetime.now() + timedelta(seconds=30),
            vehicle_counts={"north": 8, "south": 6, "east": 4, "west": 5},
            queue_lengths={"north": 20.0, "south": 15.0, "east": 10.0, "west": 13.0},
            wait_times={"north": 35.0, "south": 30.0, "east": 20.0, "west": 25.0},
            signal_phase="east_west",
            prediction_confidence=0.8
        )
        
        # Use an action from the agent's action space
        action_adjustments = agent.actions[1]  # Use second action (not no-change)
        action = SignalAction(
            intersection_id="test_intersection",
            phase_adjustments=action_adjustments,
            reasoning="Test action"
        )
        
        # Get initial Q-value
        state_str = agent._discretize_state(traffic_state)
        if state_str not in agent.q_table:
            agent.q_table[state_str] = {i: 0.0 for i in range(len(agent.actions))}
        
        action_index = 1  # Use the same action index
        initial_q = agent.q_table[state_str][action_index]
        
        # Update Q-values
        reward = 10.0
        agent.update_q_values(traffic_state, action, reward, next_state)
        
        # Q-value should have changed
        updated_q = agent.q_table[state_str][action_index]
        assert updated_q != initial_q
    
    def test_training_episode(self, agent):
        """Test training on an episode."""
        # Create sequence of traffic states
        states = []
        base_time = datetime.now()
        
        for i in range(5):
            state = TrafficState(
                intersection_id="test_intersection",
                timestamp=base_time + timedelta(seconds=i*30),
                vehicle_counts={"north": 10-i, "south": 8-i, "east": 5, "west": 7},
                queue_lengths={"north": 25.0-i*2, "south": 20.0-i*2, "east": 15.0, "west": 18.0},
                wait_times={"north": 45.0-i*3, "south": 40.0-i*3, "east": 30.0, "west": 35.0},
                signal_phase="north_south" if i % 2 == 0 else "east_west",
                prediction_confidence=0.8
            )
            states.append(state)
        
        initial_episodes = agent.training_episodes
        reward = agent.train_episode(states)
        
        assert isinstance(reward, float)
        assert agent.training_episodes == initial_episodes + 1
        assert len(agent.episode_rewards) > 0
        assert len(agent.q_table) > 0
    
    def test_training_episode_insufficient_states(self, agent):
        """Test training with insufficient states."""
        states = [TrafficState(
            intersection_id="test_intersection",
            timestamp=datetime.now(),
            vehicle_counts={"north": 10, "south": 8, "east": 5, "west": 7},
            queue_lengths={"north": 25.0, "south": 20.0, "east": 15.0, "west": 18.0},
            wait_times={"north": 45.0, "south": 40.0, "east": 30.0, "west": 35.0},
            signal_phase="north_south",
            prediction_confidence=0.8
        )]
        
        reward = agent.train_episode(states)
        assert reward == 0.0
    
    def test_epsilon_decay(self, agent):
        """Test epsilon decay during training."""
        initial_epsilon = agent.config.epsilon
        
        # Create dummy states for training
        states = []
        base_time = datetime.now()
        
        for i in range(3):
            state = TrafficState(
                intersection_id="test_intersection",
                timestamp=base_time + timedelta(seconds=i*30),
                vehicle_counts={"north": 10, "south": 8, "east": 5, "west": 7},
                queue_lengths={"north": 25.0, "south": 20.0, "east": 15.0, "west": 18.0},
                wait_times={"north": 45.0, "south": 40.0, "east": 30.0, "west": 35.0},
                signal_phase="north_south",
                prediction_confidence=0.8
            )
            states.append(state)
        
        agent.train_episode(states)
        
        # Epsilon should have decayed
        assert agent.config.epsilon < initial_epsilon
        assert agent.config.epsilon >= agent.config.epsilon_min
    
    def test_training_statistics(self, agent):
        """Test training statistics."""
        stats = agent.get_training_stats()
        
        assert isinstance(stats, dict)
        assert 'episodes' in stats
        assert 'total_reward' in stats
        assert 'average_reward' in stats
        assert 'epsilon' in stats
        assert 'q_table_size' in stats
        assert 'recent_rewards' in stats
        
        assert stats['episodes'] == 0
        assert stats['total_reward'] == 0.0
        assert stats['q_table_size'] == 0
    
    def test_model_save_load(self, agent, traffic_state):
        """Test model saving and loading."""
        # Train agent a bit
        states = [traffic_state, traffic_state]  # Minimal training
        agent.train_episode(states)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            agent.save_model(temp_path)
            assert os.path.exists(temp_path)
            
            # Create new agent and load model
            new_agent = QLearningAgent("new_intersection", ["phase1", "phase2"])
            new_agent.load_model(temp_path)
            
            # Check that data was loaded correctly
            assert new_agent.intersection_id == agent.intersection_id
            assert new_agent.signal_phases == agent.signal_phases
            assert new_agent.training_episodes == agent.training_episodes
            assert new_agent.q_table == agent.q_table
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_reasoning_generation(self, agent, traffic_state):
        """Test reasoning generation for actions."""
        # Test no adjustment
        no_adjustment = {"north_south": 0, "east_west": 0}
        reasoning = agent._generate_reasoning(traffic_state, no_adjustment)
        assert "No adjustment needed" in reasoning
        
        # Test single phase adjustment
        single_adjustment = {"north_south": 10, "east_west": 0}
        reasoning = agent._generate_reasoning(traffic_state, single_adjustment)
        assert "Extending north_south phase by 10s" in reasoning
        
        # Test reduction
        reduction = {"north_south": -5, "east_west": 0}
        reasoning = agent._generate_reasoning(traffic_state, reduction)
        assert "Reducing north_south phase by 5s" in reasoning
        
        # Test multiple adjustments
        multiple = {"north_south": 5, "east_west": -10}
        reasoning = agent._generate_reasoning(traffic_state, multiple)
        assert "Adjusting multiple phases" in reasoning