"""Tests for the LSTM-based traffic prediction engine."""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from src.processors.prediction_engine import (
    PredictionEngine, TrafficLSTM, PredictionResult
)
from src.models.traffic_state import TrafficState


class TestTrafficLSTM:
    """Test cases for the TrafficLSTM neural network."""
    
    def test_lstm_initialization(self):
        """Test LSTM model initialization."""
        model = TrafficLSTM(input_size=7, hidden_size=32, num_layers=2)
        
        assert model.hidden_size == 32
        assert model.num_layers == 2
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert isinstance(model.fc, torch.nn.Linear)
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        model = TrafficLSTM(input_size=7, hidden_size=32, num_layers=2, output_size=6)
        
        # Create sample input (batch_size=2, sequence_length=12, features=7)
        x = torch.randn(2, 12, 7)
        
        output = model(x)
        
        assert output.shape == (2, 6)  # batch_size, output_size
        assert not torch.isnan(output).any()
    
    def test_lstm_different_configurations(self):
        """Test LSTM with different configurations."""
        configs = [
            {'input_size': 5, 'hidden_size': 16, 'num_layers': 1, 'output_size': 1},
            {'input_size': 10, 'hidden_size': 64, 'num_layers': 3, 'output_size': 12},
        ]
        
        for config in configs:
            model = TrafficLSTM(**config)
            x = torch.randn(1, 10, config['input_size'])
            output = model(x)
            assert output.shape == (1, config['output_size'])


class TestPredictionEngine:
    """Test cases for the PredictionEngine."""
    
    @pytest.fixture
    def sample_traffic_states(self):
        """Create sample traffic states for testing."""
        states = []
        base_time = datetime(2024, 1, 1, 8, 0, 0)
        
        for i in range(50):  # 50 time steps
            timestamp = base_time + timedelta(minutes=i * 5)
            
            # Simulate realistic traffic patterns
            hour = timestamp.hour
            if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                base_vehicles = 25 + np.random.randint(-5, 6)
                base_queue = 40 + np.random.randint(-10, 11)
                base_wait = 80 + np.random.randint(-20, 21)
            else:
                base_vehicles = 10 + np.random.randint(-3, 4)
                base_queue = 15 + np.random.randint(-5, 6)
                base_wait = 30 + np.random.randint(-10, 11)
            
            state = TrafficState(
                intersection_id="test_intersection",
                timestamp=timestamp,
                vehicle_counts={"north": base_vehicles // 2, "south": base_vehicles // 2},
                queue_lengths={"north": base_queue / 2, "south": base_queue / 2},
                wait_times={"north": base_wait, "south": base_wait},
                signal_phase="north_south",
                prediction_confidence=0.8
            )
            states.append(state)
        
        return states
    
    @pytest.fixture
    def prediction_engine(self):
        """Create a prediction engine for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")
            engine = PredictionEngine(
                sequence_length=12,
                prediction_horizon=6,
                model_save_path=model_path
            )
            yield engine
    
    def test_prediction_engine_initialization(self, prediction_engine):
        """Test prediction engine initialization."""
        assert prediction_engine.sequence_length == 12
        assert prediction_engine.prediction_horizon == 6
        assert not prediction_engine.is_trained
        assert prediction_engine.model is None
        assert len(prediction_engine.feature_columns) == 7
    
    def test_prepare_features(self, prediction_engine, sample_traffic_states):
        """Test feature preparation from traffic states."""
        df = prediction_engine.prepare_features(sample_traffic_states[:10])
        
        assert len(df) == 10
        assert 'total_vehicles' in df.columns
        assert 'total_queue_length' in df.columns
        assert 'average_wait_time' in df.columns
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'is_weekend' in df.columns
        assert 'is_rush_hour' in df.columns
        
        # Check feature engineering
        assert df['is_weekend'].dtype == int
        assert df['is_rush_hour'].dtype == int
        assert all(df['hour'] >= 0) and all(df['hour'] <= 23)
        assert all(df['day_of_week'] >= 0) and all(df['day_of_week'] <= 6)
    
    def test_prepare_features_empty_input(self, prediction_engine):
        """Test feature preparation with empty input."""
        with pytest.raises(ValueError, match="No traffic states provided"):
            prediction_engine.prepare_features([])
    
    def test_create_sequences(self, prediction_engine, sample_traffic_states):
        """Test sequence creation for LSTM training."""
        df = prediction_engine.prepare_features(sample_traffic_states)
        X, y = prediction_engine.create_sequences(df)
        
        expected_sequences = len(df) - prediction_engine.sequence_length - prediction_engine.prediction_horizon + 1
        
        assert X.shape[0] == expected_sequences
        assert X.shape[1] == prediction_engine.sequence_length
        assert X.shape[2] == len(prediction_engine.feature_columns)
        assert y.shape[0] == expected_sequences
        assert y.shape[1] == prediction_engine.prediction_horizon
    
    def test_create_sequences_insufficient_data(self, prediction_engine):
        """Test sequence creation with insufficient data."""
        # Create minimal traffic states
        states = []
        for i in range(5):  # Less than sequence_length + prediction_horizon
            state = TrafficState(
                intersection_id="test",
                timestamp=datetime.now() + timedelta(minutes=i*5),
                vehicle_counts={"north": 10},
                queue_lengths={"north": 20.0},
                wait_times={"north": 30.0},
                signal_phase="north_south"
            )
            states.append(state)
        
        df = prediction_engine.prepare_features(states)
        
        with pytest.raises(ValueError, match="Not enough data points"):
            prediction_engine.create_sequences(df)
    
    def test_model_training(self, prediction_engine, sample_traffic_states):
        """Test model training process."""
        # Train with sample data
        history = prediction_engine.train_model(
            sample_traffic_states,
            epochs=5,  # Small number for testing
            batch_size=4
        )
        
        assert prediction_engine.is_trained
        assert prediction_engine.model is not None
        assert prediction_engine.scaler is not None
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
    
    def test_model_training_insufficient_data(self, prediction_engine):
        """Test model training with insufficient data."""
        # Create minimal data
        states = []
        for i in range(10):  # Less than needed for proper training
            state = TrafficState(
                intersection_id="test",
                timestamp=datetime.now() + timedelta(minutes=i*5),
                vehicle_counts={"north": 10},
                queue_lengths={"north": 20.0},
                wait_times={"north": 30.0},
                signal_phase="north_south"
            )
            states.append(state)
        
        with pytest.raises(ValueError, match="Not enough data points"):
            prediction_engine.train_model(states, epochs=1)
    
    def test_prediction_without_training(self, prediction_engine, sample_traffic_states):
        """Test prediction without training the model."""
        with pytest.raises(ValueError, match="Model must be trained"):
            prediction_engine.predict_traffic_volume(
                "test_intersection",
                sample_traffic_states[:12]
            )
    
    def test_prediction_insufficient_states(self, prediction_engine, sample_traffic_states):
        """Test prediction with insufficient recent states."""
        # Train the model first
        prediction_engine.train_model(sample_traffic_states, epochs=2)
        
        # Try to predict with insufficient states
        with pytest.raises(ValueError, match="Need at least"):
            prediction_engine.predict_traffic_volume(
                "test_intersection",
                sample_traffic_states[:5]  # Less than sequence_length
            )
    
    def test_successful_prediction(self, prediction_engine, sample_traffic_states):
        """Test successful traffic prediction."""
        # Train the model
        prediction_engine.train_model(sample_traffic_states, epochs=3)
        
        # Make prediction
        result = prediction_engine.predict_traffic_volume(
            "test_intersection",
            sample_traffic_states[-15:],  # Use last 15 states
            horizon_minutes=30
        )
        
        assert isinstance(result, PredictionResult)
        assert result.intersection_id == "test_intersection"
        assert len(result.predictions) == prediction_engine.prediction_horizon
        assert len(result.timestamps) == prediction_engine.prediction_horizon
        assert 0.0 <= result.confidence <= 1.0
        assert result.horizon_minutes == 30
        assert all(pred >= 0 for pred in result.predictions)  # Non-negative predictions
    
    def test_prediction_confidence_calculation(self, prediction_engine, sample_traffic_states):
        """Test prediction confidence calculation."""
        # Test with high-quality states
        high_quality_states = []
        for state in sample_traffic_states[:15]:
            state.prediction_confidence = 0.9
            high_quality_states.append(state)
        
        confidence = prediction_engine._calculate_prediction_confidence(high_quality_states)
        assert confidence > 0.7
        
        # Test with low-quality states
        low_quality_states = []
        for state in sample_traffic_states[:15]:
            state.prediction_confidence = 0.3
            low_quality_states.append(state)
        
        confidence = prediction_engine._calculate_prediction_confidence(low_quality_states)
        assert confidence < 0.5
    
    def test_confidence_thresholds(self, prediction_engine):
        """Test confidence threshold methods."""
        assert prediction_engine.is_high_confidence(0.9)
        assert not prediction_engine.is_high_confidence(0.7)
        
        assert prediction_engine.is_low_confidence(0.3)
        assert not prediction_engine.is_low_confidence(0.7)
    
    def test_fallback_prediction(self, prediction_engine, sample_traffic_states):
        """Test fallback prediction mechanism."""
        result = prediction_engine.get_fallback_prediction(
            "test_intersection",
            sample_traffic_states[-10:],
            horizon_minutes=30
        )
        
        assert isinstance(result, PredictionResult)
        assert result.intersection_id == "test_intersection"
        assert len(result.predictions) > 0
        assert len(result.timestamps) > 0
        assert result.confidence < 0.5  # Fallback should have low confidence
        assert 'fallback' in result.features_used or 'moving_average' in result.features_used
    
    def test_fallback_prediction_no_data(self, prediction_engine):
        """Test fallback prediction with no historical data."""
        result = prediction_engine.get_fallback_prediction(
            "test_intersection",
            [],
            horizon_minutes=30
        )
        
        assert isinstance(result, PredictionResult)
        assert all(pred == 0.0 for pred in result.predictions)
        assert result.confidence == 0.0
    
    def test_fallback_prediction_rush_hour(self, prediction_engine):
        """Test fallback prediction during rush hour."""
        # Create rush hour state
        rush_hour_state = TrafficState(
            intersection_id="test",
            timestamp=datetime(2024, 1, 1, 8, 30, 0),  # Rush hour
            vehicle_counts={"north": 15, "south": 15},
            queue_lengths={"north": 25.0, "south": 25.0},
            wait_times={"north": 60.0, "south": 60.0},
            signal_phase="north_south"
        )
        
        result = prediction_engine.get_fallback_prediction(
            "test_intersection",
            [rush_hour_state],
            horizon_minutes=15
        )
        
        # Should predict higher traffic during rush hour
        assert all(pred > 30 for pred in result.predictions)  # Rush hour multiplier applied
    
    def test_model_save_and_load(self, prediction_engine, sample_traffic_states):
        """Test model saving and loading."""
        # Train and save model
        prediction_engine.train_model(sample_traffic_states, epochs=2)
        original_predictions = prediction_engine.predict_traffic_volume(
            "test_intersection",
            sample_traffic_states[-15:]
        )
        
        prediction_engine.save_model()
        
        # Create new engine and load model
        new_engine = PredictionEngine(
            sequence_length=12,
            prediction_horizon=6,
            model_save_path=prediction_engine.model_save_path
        )
        
        assert new_engine.load_model()
        assert new_engine.is_trained
        
        # Test that loaded model produces same predictions
        loaded_predictions = new_engine.predict_traffic_volume(
            "test_intersection",
            sample_traffic_states[-15:]
        )
        
        np.testing.assert_array_almost_equal(
            original_predictions.predictions,
            loaded_predictions.predictions,
            decimal=5
        )
    
    def test_model_load_nonexistent_file(self, prediction_engine):
        """Test loading model from nonexistent file."""
        prediction_engine.model_save_path = "nonexistent_model.pth"
        assert not prediction_engine.load_model()
        assert not prediction_engine.is_trained
    
    @patch('src.processors.prediction_engine.torch.save')
    def test_model_save_error_handling(self, mock_save, prediction_engine, sample_traffic_states):
        """Test error handling during model saving."""
        prediction_engine.train_model(sample_traffic_states, epochs=1)
        
        # Mock save to raise an exception
        mock_save.side_effect = Exception("Save failed")
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            prediction_engine.save_model()
    
    def test_prediction_with_different_horizons(self, prediction_engine, sample_traffic_states):
        """Test predictions with different time horizons."""
        prediction_engine.train_model(sample_traffic_states, epochs=2)
        
        horizons = [15, 30, 60]
        for horizon in horizons:
            result = prediction_engine.predict_traffic_volume(
                "test_intersection",
                sample_traffic_states[-15:],
                horizon_minutes=horizon
            )
            
            assert result.horizon_minutes == horizon
            assert len(result.predictions) == prediction_engine.prediction_horizon
            assert len(result.timestamps) == prediction_engine.prediction_horizon
    
    def test_feature_columns_consistency(self, prediction_engine):
        """Test that feature columns are consistent throughout the process."""
        expected_features = [
            'total_vehicles', 'total_queue_length', 'average_wait_time',
            'hour', 'day_of_week', 'is_weekend', 'is_rush_hour'
        ]
        
        assert prediction_engine.feature_columns == expected_features
    
    def test_prediction_result_dataclass(self):
        """Test PredictionResult dataclass."""
        result = PredictionResult(
            intersection_id="test",
            predictions=[10.0, 15.0, 20.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(3)],
            confidence=0.85,
            horizon_minutes=15,
            features_used=['test_feature']
        )
        
        assert result.intersection_id == "test"
        assert len(result.predictions) == 3
        assert len(result.timestamps) == 3
        assert result.confidence == 0.85
        assert result.horizon_minutes == 15
        assert result.features_used == ['test_feature']


class TestPredictionEngineIntegration:
    """Integration tests for the prediction engine."""
    
    def test_end_to_end_prediction_workflow(self):
        """Test complete prediction workflow from training to prediction."""
        # Create realistic traffic data
        states = []
        base_time = datetime(2024, 1, 1, 6, 0, 0)
        
        for i in range(100):  # More data for better training
            timestamp = base_time + timedelta(minutes=i * 5)
            hour = timestamp.hour
            
            # Realistic traffic patterns
            if 6 <= hour <= 9 or 17 <= hour <= 20:  # Rush hours
                vehicles = np.random.poisson(30)
                queue = np.random.exponential(50)
                wait = np.random.exponential(90)
            elif 22 <= hour or hour <= 5:  # Night
                vehicles = np.random.poisson(5)
                queue = np.random.exponential(10)
                wait = np.random.exponential(20)
            else:  # Regular hours
                vehicles = np.random.poisson(15)
                queue = np.random.exponential(25)
                wait = np.random.exponential(45)
            
            state = TrafficState(
                intersection_id="main_intersection",
                timestamp=timestamp,
                vehicle_counts={"north": vehicles // 2, "south": vehicles // 2},
                queue_lengths={"north": queue / 2, "south": queue / 2},
                wait_times={"north": wait, "south": wait},
                signal_phase="north_south",
                prediction_confidence=0.8 + np.random.normal(0, 0.1)
            )
            states.append(state)
        
        # Initialize and train prediction engine
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "integration_test_model.pth")
            engine = PredictionEngine(model_save_path=model_path)
            
            # Train model
            history = engine.train_model(states, epochs=10, batch_size=8)
            
            # Verify training
            assert engine.is_trained
            assert len(history['train_loss']) == 10
            assert history['train_loss'][-1] < history['train_loss'][0]  # Loss should decrease
            
            # Make predictions
            recent_states = states[-20:]
            result = engine.predict_traffic_volume(
                "main_intersection",
                recent_states,
                horizon_minutes=30
            )
            
            # Verify predictions
            assert isinstance(result, PredictionResult)
            assert result.confidence > 0.0
            assert len(result.predictions) == engine.prediction_horizon
            assert all(pred >= 0 for pred in result.predictions)
            
            # Test model persistence
            engine.save_model()
            
            # Load in new engine
            new_engine = PredictionEngine(model_save_path=model_path)
            assert new_engine.load_model()
            
            # Verify loaded model works
            new_result = new_engine.predict_traffic_volume(
                "main_intersection",
                recent_states,
                horizon_minutes=30
            )
            
            np.testing.assert_array_almost_equal(
                result.predictions,
                new_result.predictions,
                decimal=5
            )