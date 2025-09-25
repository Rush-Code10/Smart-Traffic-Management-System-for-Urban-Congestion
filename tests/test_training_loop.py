"""Tests for training loop and traffic scenario generator."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.agents.training_loop import (
    TrainingLoop, TrafficScenarioGenerator, TrainingConfig
)
from src.agents.q_learning_agent import QLearningAgent, QLearningConfig
from src.config.intersection_config import IntersectionConfig
from src.models.traffic_state import TrafficState


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.max_episodes == 1000
        assert config.episode_length == 300
        assert config.time_step == 5
        assert config.convergence_threshold == 0.01
        assert config.convergence_window == 50
        assert config.save_interval == 100
        assert config.log_interval == 10
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = TrainingConfig(max_episodes=100, episode_length=200)
        config.validate()
        
        # Invalid max_episodes
        with pytest.raises(ValueError, match="max_episodes must be positive"):
            TrainingConfig(max_episodes=0).validate()
        
        # Invalid episode_length
        with pytest.raises(ValueError, match="episode_length must be positive"):
            TrainingConfig(episode_length=-1).validate()
        
        # Invalid time_step
        with pytest.raises(ValueError, match="time_step must be positive"):
            TrainingConfig(time_step=0).validate()
        
        # Invalid convergence_threshold
        with pytest.raises(ValueError, match="convergence_threshold must be between 0 and 1"):
            TrainingConfig(convergence_threshold=0.0).validate()
        
        with pytest.raises(ValueError, match="convergence_threshold must be between 0 and 1"):
            TrainingConfig(convergence_threshold=1.5).validate()
        
        # Invalid convergence_window
        with pytest.raises(ValueError, match="convergence_window must be positive"):
            TrainingConfig(convergence_window=0).validate()


class TestTrafficScenarioGenerator:
    """Test traffic scenario generator."""
    
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
    def scenario_generator(self, intersection_config):
        """Create scenario generator for testing."""
        return TrafficScenarioGenerator(intersection_config)
    
    def test_initialization(self, scenario_generator, intersection_config):
        """Test scenario generator initialization."""
        assert scenario_generator.intersection_config == intersection_config
        assert scenario_generator.directions == ["north", "south", "east", "west"]
        assert len(scenario_generator.scenarios) > 0
        assert 'light' in scenario_generator.scenarios
        assert 'heavy' in scenario_generator.scenarios
    
    def test_generate_traffic_sequence(self, scenario_generator):
        """Test traffic sequence generation."""
        states = scenario_generator.generate_traffic_sequence(
            scenario='moderate',
            duration=60,
            time_step=10
        )
        
        assert len(states) == 6  # 60 seconds / 10 second steps
        
        for state in states:
            assert isinstance(state, TrafficState)
            assert state.intersection_id == "test_intersection"
            assert len(state.vehicle_counts) == 4  # Four directions
            assert len(state.queue_lengths) == 4
            assert len(state.wait_times) == 4
            assert state.signal_phase in ["north_south", "east_west"]
            
            # Check that values are reasonable
            for direction in ["north", "south", "east", "west"]:
                assert direction in state.vehicle_counts
                assert state.vehicle_counts[direction] >= 0
                assert state.queue_lengths[direction] >= 0
                assert state.wait_times[direction] >= 0
    
    def test_generate_traffic_sequence_unknown_scenario(self, scenario_generator):
        """Test generation with unknown scenario."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            scenario_generator.generate_traffic_sequence(
                scenario='unknown_scenario',
                duration=60,
                time_step=10
            )
    
    def test_different_scenarios(self, scenario_generator):
        """Test that different scenarios produce different traffic patterns."""
        light_states = scenario_generator.generate_traffic_sequence('light', 60, 20)
        heavy_states = scenario_generator.generate_traffic_sequence('heavy', 60, 20)
        
        # Heavy traffic should generally have more vehicles
        light_avg = np.mean([s.get_total_vehicles() for s in light_states])
        heavy_avg = np.mean([s.get_total_vehicles() for s in heavy_states])
        
        assert heavy_avg > light_avg
    
    def test_asymmetric_scenario(self, scenario_generator):
        """Test asymmetric traffic scenario."""
        states = scenario_generator.generate_traffic_sequence('asymmetric', 60, 20)
        
        # Check that traffic is indeed asymmetric
        total_north = sum(s.vehicle_counts['north'] for s in states)
        total_south = sum(s.vehicle_counts['south'] for s in states)
        
        # North should have more traffic than south in asymmetric scenario
        assert total_north > total_south
    
    def test_direction_factor(self, scenario_generator):
        """Test direction factor calculation."""
        # Asymmetric scenario should have different factors
        north_factor = scenario_generator._get_direction_factor('north', 'asymmetric')
        south_factor = scenario_generator._get_direction_factor('south', 'asymmetric')
        
        assert north_factor != south_factor
        
        # Other scenarios should have uniform factors
        light_north = scenario_generator._get_direction_factor('north', 'light')
        light_south = scenario_generator._get_direction_factor('south', 'light')
        
        assert light_north == light_south == 1.0


class TestTrainingLoop:
    """Test training loop."""
    
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
    def training_loop(self, intersection_config):
        """Create training loop for testing."""
        agent_config = QLearningConfig(epsilon=0.5, max_adjustment=10)
        training_config = TrainingConfig(max_episodes=10, episode_length=60, time_step=20)
        
        return TrainingLoop(
            intersection_config=intersection_config,
            agent_config=agent_config,
            training_config=training_config
        )
    
    def test_initialization(self, training_loop, intersection_config):
        """Test training loop initialization."""
        assert training_loop.intersection_config == intersection_config
        assert isinstance(training_loop.agent, QLearningAgent)
        assert training_loop.current_episode == 0
        assert len(training_loop.episode_rewards) == 0
        assert not training_loop.convergence_achieved
    
    def test_set_callbacks(self, training_loop):
        """Test setting callback functions."""
        episode_callback = Mock()
        convergence_callback = Mock()
        
        training_loop.set_episode_callback(episode_callback)
        training_loop.set_convergence_callback(convergence_callback)
        
        assert training_loop.episode_callback == episode_callback
        assert training_loop.convergence_callback == convergence_callback
    
    def test_run_episode(self, training_loop):
        """Test running a single episode."""
        initial_episodes = training_loop.agent.training_episodes
        
        reward = training_loop._run_episode('moderate')
        
        assert isinstance(reward, float)
        assert training_loop.agent.training_episodes == initial_episodes + 1
    
    def test_convergence_check_insufficient_data(self, training_loop):
        """Test convergence check with insufficient data."""
        # Not enough episodes
        training_loop.episode_rewards = [1.0, 2.0, 3.0]
        assert not training_loop._check_convergence()
    
    def test_convergence_check_converged(self, training_loop):
        """Test convergence check when converged."""
        # Create stable rewards (low variation)
        stable_rewards = [10.0] * training_loop.training_config.convergence_window
        training_loop.episode_rewards = stable_rewards
        
        assert training_loop._check_convergence()
    
    def test_convergence_check_not_converged(self, training_loop):
        """Test convergence check when not converged."""
        # Create highly variable rewards
        variable_rewards = list(range(training_loop.training_config.convergence_window))
        training_loop.episode_rewards = variable_rewards
        
        assert not training_loop._check_convergence()
    
    def test_training_stats(self, training_loop):
        """Test training statistics."""
        # Add some episode rewards
        training_loop.episode_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        training_loop.current_episode = 5
        
        stats = training_loop._get_training_stats()
        
        assert isinstance(stats, dict)
        assert stats['episode'] == 5
        assert stats['total_episodes'] == 5
        assert stats['avg_reward_total'] == 3.0  # Mean of [1,2,3,4,5]
        assert 'epsilon' in stats
        assert 'q_table_size' in stats
        assert 'convergence_achieved' in stats
    
    def test_train_with_early_convergence(self, training_loop):
        """Test training that converges early."""
        # Mock convergence check to return True after a few episodes
        with patch.object(training_loop, '_check_convergence') as mock_convergence:
            mock_convergence.side_effect = [False, False, True]  # Converge on 3rd episode
            
            convergence_callback = Mock()
            training_loop.set_convergence_callback(convergence_callback)
            
            results = training_loop.train(scenarios=['light'])
            
            assert results['success'] is True
            assert results['convergence_achieved'] is True
            assert results['episodes_completed'] == 3
            convergence_callback.assert_called_once_with(3)
    
    def test_train_full_episodes(self, training_loop):
        """Test training for full episode count."""
        # Use very small episode count for fast test
        training_loop.training_config.max_episodes = 3
        
        episode_callback = Mock()
        training_loop.set_episode_callback(episode_callback)
        
        results = training_loop.train(scenarios=['light'])
        
        assert results['success'] is True
        assert results['episodes_completed'] == 3
        assert episode_callback.call_count == 3
    
    def test_train_with_model_saving(self, training_loop):
        """Test training with model saving."""
        training_loop.training_config.max_episodes = 2
        training_loop.training_config.save_interval = 1  # Save every episode
        
        with patch.object(training_loop.agent, 'save_model') as mock_save:
            results = training_loop.train(scenarios=['light'], model_save_path='test_model')
            
            # Should save during training and final save
            assert mock_save.call_count >= 2
            
            # Check that final save was called
            final_call = mock_save.call_args_list[-1]
            assert 'test_model_final.pkl' in final_call[0][0]
    
    def test_train_keyboard_interrupt(self, training_loop):
        """Test training interruption."""
        with patch.object(training_loop, '_run_episode') as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            
            with patch.object(training_loop.agent, 'save_model') as mock_save:
                results = training_loop.train(model_save_path='test_model')
                
                # Should still save final model
                mock_save.assert_called()
    
    def test_evaluate_agent(self, training_loop):
        """Test agent evaluation."""
        # Train agent a bit first
        training_loop.train(scenarios=['light'])
        
        # Evaluate
        results = training_loop.evaluate_agent('moderate', num_episodes=3)
        
        assert isinstance(results, dict)
        assert results['scenario'] == 'moderate'
        assert results['num_episodes'] == 3
        assert 'average_reward' in results
        assert 'reward_std' in results
        assert 'best_reward' in results
        assert 'worst_reward' in results
        assert len(results['all_rewards']) == 3
    
    def test_evaluate_agent_no_exploration(self, training_loop):
        """Test that evaluation disables exploration."""
        original_epsilon = training_loop.agent.config.epsilon
        
        with patch.object(training_loop, '_run_episode') as mock_run:
            mock_run.return_value = 10.0
            
            training_loop.evaluate_agent('light', num_episodes=1)
            
            # During evaluation, epsilon should be 0
            # After evaluation, epsilon should be restored
            assert training_loop.agent.config.epsilon == original_epsilon
    
    def test_training_results_format(self, training_loop):
        """Test training results format."""
        training_loop.training_config.max_episodes = 2
        results = training_loop.train(scenarios=['light'])
        
        # Check all expected keys are present
        expected_keys = [
            'success', 'episodes_completed', 'convergence_achieved',
            'final_epsilon', 'q_table_size', 'total_reward',
            'average_reward', 'best_episode_reward', 'worst_episode_reward',
            'reward_std', 'training_time', 'episodes_per_second'
        ]
        
        for key in expected_keys:
            assert key in results
        
        assert results['success'] is True
        assert results['episodes_completed'] == 2
        assert isinstance(results['training_time'], float)
        assert results['training_time'] > 0