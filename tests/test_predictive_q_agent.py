"""Tests for the predictive Q-learning agent."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from src.agents.predictive_q_agent import PredictiveQLearningAgent, PredictiveConfig
from src.processors.prediction_engine import PredictionEngine, PredictionResult
from src.models.traffic_state import TrafficState
from src.models.signal_action import SignalAction


class TestPredictiveConfig:
    """Test cases for PredictiveConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PredictiveConfig()
        
        assert config.prediction_weight == 0.3
        assert config.prediction_horizon == 30
        assert config.confidence_threshold == 0.6
        assert config.proactive_adjustment_factor == 1.5
        
        # Should also have base QLearningConfig attributes
        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'discount_factor')
        assert hasattr(config, 'epsilon')
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = PredictiveConfig(
            prediction_weight=0.5,
            prediction_horizon=45,
            confidence_threshold=0.7,
            proactive_adjustment_factor=2.0
        )
        config.validate()  # Should not raise
        
        # Invalid prediction_weight
        with pytest.raises(ValueError, match="prediction_weight must be between 0 and 1"):
            config = PredictiveConfig(prediction_weight=1.5)
            config.validate()
        
        # Invalid prediction_horizon
        with pytest.raises(ValueError, match="prediction_horizon must be a positive integer"):
            config = PredictiveConfig(prediction_horizon=-10)
            config.validate()
        
        # Invalid confidence_threshold
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            config = PredictiveConfig(confidence_threshold=1.2)
            config.validate()
        
        # Invalid proactive_adjustment_factor
        with pytest.raises(ValueError, match="proactive_adjustment_factor must be a positive number"):
            config = PredictiveConfig(proactive_adjustment_factor=-1.0)
            config.validate()


class TestPredictiveQLearningAgent:
    """Test cases for PredictiveQLearningAgent."""
    
    @pytest.fixture
    def mock_prediction_engine(self):
        """Create a mock prediction engine."""
        engine = Mock(spec=PredictionEngine)
        engine.sequence_length = 12
        engine.is_trained = True
        return engine
    
    @pytest.fixture
    def sample_traffic_states(self):
        """Create sample traffic states for testing."""
        states = []
        base_time = datetime(2024, 1, 1, 8, 0, 0)
        
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i * 5)
            vehicles = 15 + np.random.randint(-5, 6)
            
            state = TrafficState(
                intersection_id="test_intersection",
                timestamp=timestamp,
                vehicle_counts={"north": vehicles // 2, "south": vehicles // 2},
                queue_lengths={"north": 20.0, "south": 25.0},
                wait_times={"north": 45.0, "south": 50.0},
                signal_phase="north_south",
                prediction_confidence=0.8
            )
            states.append(state)
        
        return states
    
    @pytest.fixture
    def predictive_agent(self, mock_prediction_engine):
        """Create a predictive Q-learning agent for testing."""
        config = PredictiveConfig(
            learning_rate=0.1,
            epsilon=0.1,
            prediction_weight=0.3
        )
        
        agent = PredictiveQLearningAgent(
            intersection_id="test_intersection",
            signal_phases=["north_south", "east_west"],
            prediction_engine=mock_prediction_engine,
            config=config
        )
        
        return agent
    
    def test_agent_initialization(self, predictive_agent, mock_prediction_engine):
        """Test predictive agent initialization."""
        assert predictive_agent.intersection_id == "test_intersection"
        assert predictive_agent.signal_phases == ["north_south", "east_west"]
        assert predictive_agent.prediction_engine == mock_prediction_engine
        assert isinstance(predictive_agent.predictive_config, PredictiveConfig)
        assert predictive_agent.recent_states == []
        assert predictive_agent.prediction_history == []
        assert predictive_agent.proactive_actions_taken == 0
    
    def test_update_recent_states(self, predictive_agent, sample_traffic_states):
        """Test updating recent states."""
        # Add states one by one
        for i, state in enumerate(sample_traffic_states[:5]):
            predictive_agent.update_recent_states(state)
            assert len(predictive_agent.recent_states) == i + 1
        
        # Test that old states are removed when limit is reached
        for state in sample_traffic_states[5:15]:  # Add 10 more states
            predictive_agent.update_recent_states(state)
        
        assert len(predictive_agent.recent_states) <= 12  # Should not exceed max_states
    
    def test_get_traffic_prediction_insufficient_states(self, predictive_agent, sample_traffic_states):
        """Test prediction with insufficient recent states."""
        # Add only a few states
        for state in sample_traffic_states[:5]:
            predictive_agent.update_recent_states(state)
        
        prediction = predictive_agent.get_traffic_prediction()
        assert prediction is None
    
    def test_get_traffic_prediction_success(self, predictive_agent, sample_traffic_states):
        """Test successful traffic prediction."""
        # Add enough states
        for state in sample_traffic_states:
            predictive_agent.update_recent_states(state)
        
        # Mock the prediction engine to return a prediction
        mock_prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[20.0, 25.0, 30.0, 28.0, 22.0, 18.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(6)],
            confidence=0.85,
            horizon_minutes=30,
            features_used=['test_features']
        )
        
        predictive_agent.prediction_engine.predict_traffic_volume.return_value = mock_prediction
        
        prediction = predictive_agent.get_traffic_prediction()
        
        assert prediction == mock_prediction
        assert len(predictive_agent.prediction_history) == 1
        predictive_agent.prediction_engine.predict_traffic_volume.assert_called_once()
    
    def test_get_traffic_prediction_exception_handling(self, predictive_agent, sample_traffic_states):
        """Test prediction with engine exception."""
        # Add enough states
        for state in sample_traffic_states:
            predictive_agent.update_recent_states(state)
        
        # Mock the prediction engine to raise an exception
        predictive_agent.prediction_engine.predict_traffic_volume.side_effect = Exception("Prediction failed")
        
        prediction = predictive_agent.get_traffic_prediction()
        assert prediction is None
    
    def test_discretize_predictive_state_no_prediction(self, predictive_agent, sample_traffic_states):
        """Test state discretization without prediction."""
        state = sample_traffic_states[0]
        
        discretized = predictive_agent._discretize_predictive_state(state, None)
        
        assert discretized.endswith("_pred_none")
        assert "v" in discretized  # Should contain vehicle count bin
        assert "q" in discretized  # Should contain queue length bin
        assert "w" in discretized  # Should contain wait time bin
    
    def test_discretize_predictive_state_with_prediction(self, predictive_agent, sample_traffic_states):
        """Test state discretization with prediction."""
        state = sample_traffic_states[0]
        
        prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[25.0, 30.0, 35.0, 32.0, 28.0, 24.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(6)],
            confidence=0.85,
            horizon_minutes=30,
            features_used=['test_features']
        )
        
        discretized = predictive_agent._discretize_predictive_state(state, prediction)
        
        assert "_pred_" in discretized
        assert discretized.endswith("_high") or discretized.endswith("_med")
        assert not discretized.endswith("_pred_none")
    
    def test_discretize_predictive_state_low_confidence(self, predictive_agent, sample_traffic_states):
        """Test state discretization with low confidence prediction."""
        state = sample_traffic_states[0]
        
        prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[20.0, 22.0, 24.0, 23.0, 21.0, 19.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(6)],
            confidence=0.3,  # Low confidence
            horizon_minutes=30,
            features_used=['test_features']
        )
        
        discretized = predictive_agent._discretize_predictive_state(state, prediction)
        
        assert discretized.endswith("_pred_none")  # Should ignore low confidence prediction
    
    def test_calculate_prediction_bonus(self, predictive_agent):
        """Test prediction bonus calculation."""
        # Test with traffic increase predicted
        action = {"north_south": 10, "east_west": 0}  # Extend north_south phase
        bonus = predictive_agent._calculate_prediction_bonus(
            current_vehicles=15,
            predicted_peak=25,
            action=action,
            confidence=0.8
        )
        
        assert bonus > 0  # Should give positive bonus for proactive action
        
        # Test with no traffic increase
        bonus_no_increase = predictive_agent._calculate_prediction_bonus(
            current_vehicles=25,
            predicted_peak=20,
            action=action,
            confidence=0.8
        )
        
        assert bonus_no_increase == 0  # No bonus when no increase predicted
        
        # Test with no extension
        action_no_extension = {"north_south": -5, "east_west": 0}
        bonus_no_extension = predictive_agent._calculate_prediction_bonus(
            current_vehicles=15,
            predicted_peak=25,
            action=action_no_extension,
            confidence=0.8
        )
        
        assert bonus_no_extension == 0  # No bonus for reducing phases when increase predicted
    
    def test_is_proactive_action(self, predictive_agent, sample_traffic_states):
        """Test proactive action detection."""
        state = sample_traffic_states[0]
        
        # High confidence prediction with traffic increase
        prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[25.0, 30.0, 35.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(3)],
            confidence=0.85,
            horizon_minutes=15,
            features_used=['test_features']
        )
        
        # Proactive action (extending phases when increase predicted)
        proactive_action = {"north_south": 15, "east_west": 0}
        assert predictive_agent._is_proactive_action(state, prediction, proactive_action)
        
        # Non-proactive action (reducing phases)
        non_proactive_action = {"north_south": -10, "east_west": 0}
        assert not predictive_agent._is_proactive_action(state, prediction, non_proactive_action)
        
        # No prediction
        assert not predictive_agent._is_proactive_action(state, None, proactive_action)
        
        # Low confidence prediction
        low_conf_prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[25.0, 30.0, 35.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(3)],
            confidence=0.3,
            horizon_minutes=15,
            features_used=['test_features']
        )
        assert not predictive_agent._is_proactive_action(state, low_conf_prediction, proactive_action)
    
    def test_get_action_with_prediction(self, predictive_agent, sample_traffic_states):
        """Test action selection with prediction."""
        # Add enough states for prediction
        for state in sample_traffic_states:
            predictive_agent.update_recent_states(state)
        
        # Mock prediction
        mock_prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[25.0, 30.0, 35.0, 32.0, 28.0, 24.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(6)],
            confidence=0.85,
            horizon_minutes=30,
            features_used=['test_features']
        )
        
        predictive_agent.prediction_engine.predict_traffic_volume.return_value = mock_prediction
        
        # Set epsilon to 0 for deterministic behavior
        predictive_agent.config.epsilon = 0.0
        
        action = predictive_agent.get_action(sample_traffic_states[-1])
        
        assert isinstance(action, SignalAction)
        assert action.intersection_id == "test_intersection"
        assert "PROACTIVE" in action.reasoning or "Prediction" in action.reasoning
    
    def test_get_action_without_prediction(self, predictive_agent, sample_traffic_states):
        """Test action selection without prediction."""
        # Mock prediction engine to return None
        predictive_agent.prediction_engine.predict_traffic_volume.return_value = None
        
        # Set epsilon to 0 for deterministic behavior
        predictive_agent.config.epsilon = 0.0
        
        action = predictive_agent.get_action(sample_traffic_states[0])
        
        assert isinstance(action, SignalAction)
        assert action.intersection_id == "test_intersection"
        assert "no reliable prediction" in action.reasoning
    
    def test_calculate_predictive_reward(self, predictive_agent, sample_traffic_states):
        """Test predictive reward calculation."""
        prev_state = sample_traffic_states[0]
        current_state = sample_traffic_states[1]
        action = {"north_south": 10, "east_west": 0}
        
        # Test with prediction
        prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[25.0, 30.0, 35.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(3)],
            confidence=0.8,
            horizon_minutes=15,
            features_used=['test_features']
        )
        
        reward_with_pred = predictive_agent._calculate_predictive_reward(
            prev_state, current_state, action, prediction
        )
        
        # Test without prediction
        reward_without_pred = predictive_agent._calculate_predictive_reward(
            prev_state, current_state, action, None
        )
        
        # Both should be valid rewards
        assert isinstance(reward_with_pred, (int, float))
        assert isinstance(reward_without_pred, (int, float))
    
    def test_train_episode_with_predictions(self, predictive_agent, sample_traffic_states):
        """Test training episode with predictions."""
        # Mock prediction engine
        mock_prediction = PredictionResult(
            intersection_id="test_intersection",
            predictions=[20.0, 25.0, 30.0, 28.0, 22.0, 18.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(6)],
            confidence=0.75,
            horizon_minutes=30,
            features_used=['test_features']
        )
        
        predictive_agent.prediction_engine.predict_traffic_volume.return_value = mock_prediction
        
        # Train on sample states
        episode_reward = predictive_agent.train_episode_with_predictions(sample_traffic_states)
        
        assert isinstance(episode_reward, (int, float))
        assert predictive_agent.training_episodes == 1
        assert len(predictive_agent.episode_rewards) == 1
        assert len(predictive_agent.recent_states) > 0
    
    def test_train_episode_insufficient_states(self, predictive_agent):
        """Test training with insufficient states."""
        single_state = TrafficState(
            intersection_id="test",
            timestamp=datetime.now(),
            vehicle_counts={"north": 10},
            queue_lengths={"north": 20.0},
            wait_times={"north": 30.0},
            signal_phase="north_south"
        )
        
        reward = predictive_agent.train_episode_with_predictions([single_state])
        assert reward == 0.0
    
    def test_get_prediction_stats(self, predictive_agent):
        """Test prediction statistics."""
        # Initially no predictions
        stats = predictive_agent.get_prediction_stats()
        assert stats['predictions_made'] == 0
        
        # Add some mock predictions to history
        for i in range(5):
            prediction = PredictionResult(
                intersection_id="test",
                predictions=[20.0, 25.0],
                timestamps=[datetime.now() + timedelta(minutes=j*5) for j in range(2)],
                confidence=0.7 + i * 0.05,
                horizon_minutes=10,
                features_used=['test']
            )
            predictive_agent.prediction_history.append(prediction)
        
        predictive_agent.proactive_actions_taken = 3
        
        stats = predictive_agent.get_prediction_stats()
        
        assert stats['predictions_made'] == 5
        assert 0.7 <= stats['average_confidence'] <= 0.9
        assert stats['proactive_actions_taken'] == 3
        assert len(stats['recent_confidences']) == 5
    
    def test_get_enhanced_training_stats(self, predictive_agent):
        """Test enhanced training statistics."""
        # Add some training history
        predictive_agent.training_episodes = 10
        predictive_agent.episode_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        predictive_agent.proactive_actions_taken = 7
        
        stats = predictive_agent.get_enhanced_training_stats()
        
        # Should include both base and prediction stats
        assert 'episodes' in stats
        assert 'predictions_made' in stats
        assert 'proactive_actions_taken' in stats
        assert stats['episodes'] == 10
        assert stats['proactive_actions_taken'] == 7
    
    def test_use_fallback_when_low_confidence(self, predictive_agent, sample_traffic_states):
        """Test fallback behavior when prediction confidence is low."""
        state = sample_traffic_states[0]
        
        # Set epsilon to 0 for deterministic behavior
        predictive_agent.config.epsilon = 0.0
        
        action = predictive_agent.use_fallback_when_low_confidence(state)
        
        assert isinstance(action, SignalAction)
        assert action.intersection_id == "test_intersection"
        # Should use base Q-learning behavior
    
    def test_prediction_history_management(self, predictive_agent, sample_traffic_states):
        """Test that prediction history is properly managed."""
        # Add enough states for predictions
        for state in sample_traffic_states:
            predictive_agent.update_recent_states(state)
        
        # Add many predictions to test history limit
        for i in range(25):
            mock_prediction = PredictionResult(
                intersection_id="test_intersection",
                predictions=[20.0 + i, 25.0 + i],
                timestamps=[datetime.now() + timedelta(minutes=j*5) for j in range(2)],
                confidence=0.8,
                horizon_minutes=10,
                features_used=['test']
            )
            predictive_agent.prediction_history.append(mock_prediction)
        
        # Should be limited to 20 predictions
        assert len(predictive_agent.prediction_history) == 20
    
    def test_integration_with_base_q_learning(self, predictive_agent, sample_traffic_states):
        """Test that predictive agent properly extends base Q-learning functionality."""
        # Test that base methods still work
        state = sample_traffic_states[0]
        
        # Test state discretization (base method)
        base_state = predictive_agent._discretize_state(state)
        assert isinstance(base_state, str)
        assert "v" in base_state and "q" in base_state and "w" in base_state
        
        # Test action generation
        action = predictive_agent.get_action(state)
        assert isinstance(action, SignalAction)
        
        # Test that Q-table is being used
        assert isinstance(predictive_agent.q_table, dict)
        
        # Test training statistics
        stats = predictive_agent.get_training_stats()
        assert 'episodes' in stats
        assert 'epsilon' in stats