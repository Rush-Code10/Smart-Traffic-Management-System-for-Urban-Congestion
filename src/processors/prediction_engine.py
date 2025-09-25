"""LSTM-based traffic prediction engine for forecasting traffic patterns."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import pickle
import os

from ..models.traffic_state import TrafficState
from ..utils.error_handling import handle_error

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of traffic prediction with confidence metrics."""
    
    intersection_id: str
    predictions: List[float]  # Traffic volume predictions for each time step
    timestamps: List[datetime]  # Corresponding timestamps
    confidence: float  # Overall prediction confidence (0.0 to 1.0)
    horizon_minutes: int  # Prediction horizon in minutes
    features_used: List[str]  # Features used for prediction


class TrafficLSTM(nn.Module):
    """LSTM neural network for traffic volume forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            output_size: Number of output predictions
            dropout: Dropout rate for regularization
        """
        super(TrafficLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM network."""
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class PredictionEngine:
    """Traffic prediction engine using LSTM neural networks."""
    
    def __init__(self, sequence_length: int = 12, prediction_horizon: int = 6,
                 model_save_path: str = "models/traffic_lstm.pth"):
        """
        Initialize the prediction engine.
        
        Args:
            sequence_length: Number of historical time steps to use for prediction
            prediction_horizon: Number of future time steps to predict
            model_save_path: Path to save/load the trained model
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model_save_path = model_save_path
        
        # Feature columns for training
        self.feature_columns = [
            'total_vehicles', 'total_queue_length', 'average_wait_time',
            'hour', 'day_of_week', 'is_weekend', 'is_rush_hour'
        ]
        
        self.model: Optional[TrafficLSTM] = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.5
        
        logger.info(f"PredictionEngine initialized with sequence_length={sequence_length}, "
                   f"prediction_horizon={prediction_horizon}")
    
    def prepare_features(self, traffic_states: List[TrafficState]) -> pd.DataFrame:
        """
        Prepare features from traffic states for model training/prediction.
        
        Args:
            traffic_states: List of historical traffic states
            
        Returns:
            DataFrame with engineered features
        """
        if not traffic_states:
            raise ValueError("No traffic states provided")
        
        data = []
        for state in traffic_states:
            features = {
                'timestamp': state.timestamp,
                'intersection_id': state.intersection_id,
                'total_vehicles': state.get_total_vehicles(),
                'total_queue_length': state.get_total_queue_length(),
                'average_wait_time': state.get_average_wait_time(),
                'prediction_confidence': state.prediction_confidence
            }
            
            # Time-based features
            features['hour'] = state.timestamp.hour
            features['day_of_week'] = state.timestamp.weekday()
            features['is_weekend'] = 1 if state.timestamp.weekday() >= 5 else 0
            features['is_rush_hour'] = 1 if features['hour'] in [7, 8, 9, 17, 18, 19] else 0
            
            data.append(features)
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        
        logger.debug(f"Prepared features for {len(df)} traffic states")
        return df
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training from time series data.
        
        Args:
            data: DataFrame with features and target variable
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if len(data) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Not enough data points. Need at least "
                           f"{self.sequence_length + self.prediction_horizon}, got {len(data)}")
        
        # Use feature columns for input
        feature_data = data[self.feature_columns].values
        target_data = data['total_vehicles'].values
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(feature_data[i:i + self.sequence_length])
            
            # Target (next prediction_horizon steps)
            y.append(target_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.debug(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
        return X, y
    
    def train_model(self, traffic_states: List[TrafficState], epochs: int = 100,
                   batch_size: int = 32, learning_rate: float = 0.001,
                   validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the LSTM model on historical traffic data.
        
        Args:
            traffic_states: Historical traffic states for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history with loss values
        """
        logger.info(f"Starting model training with {len(traffic_states)} traffic states")
        
        # Prepare features
        df = self.prepare_features(traffic_states)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        # Fit scaler on feature columns only
        feature_data = df[self.feature_columns].values
        self.scaler.fit(feature_data)
        
        # Transform features
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = self.scaler.transform(feature_data)
        
        # Create sequences
        X, y = self.create_sequences(df_scaled)
        
        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Initialize model
        input_size = len(self.feature_columns)
        self.model = TrafficLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=self.prediction_horizon
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            num_batches = max(1, len(X_train) // batch_size)
            train_loss /= num_batches
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}")
        
        self.is_trained = True
        self.training_history = {'train_loss': train_losses, 'val_loss': val_losses}
        
        # Save model
        self.save_model()
        
        logger.info("Model training completed successfully")
        return self.training_history
    
    def predict_traffic_volume(self, intersection_id: str, 
                             recent_states: List[TrafficState],
                             horizon_minutes: int = 30) -> PredictionResult:
        """
        Generate traffic volume predictions for the specified horizon.
        
        Args:
            intersection_id: ID of the intersection to predict for
            recent_states: Recent traffic states for context
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            PredictionResult with predictions and confidence
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if len(recent_states) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} recent states, "
                           f"got {len(recent_states)}")
        
        logger.debug(f"Predicting traffic for intersection {intersection_id}, "
                    f"horizon: {horizon_minutes} minutes")
        
        # Prepare features from recent states
        df = self.prepare_features(recent_states[-self.sequence_length:])
        
        # Scale features
        feature_data = df[self.feature_columns].values
        scaled_features = self.scaler.transform(feature_data)
        
        # Create input sequence
        input_sequence = torch.FloatTensor(scaled_features).unsqueeze(0)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_sequence)
            predictions = predictions.squeeze().numpy()
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        # Generate timestamps for predictions
        last_timestamp = recent_states[-1].timestamp
        prediction_interval = timedelta(minutes=horizon_minutes // self.prediction_horizon)
        timestamps = [
            last_timestamp + (i + 1) * prediction_interval 
            for i in range(self.prediction_horizon)
        ]
        
        # Calculate confidence based on recent prediction accuracy
        confidence = self._calculate_prediction_confidence(recent_states)
        
        result = PredictionResult(
            intersection_id=intersection_id,
            predictions=predictions.tolist(),
            timestamps=timestamps,
            confidence=confidence,
            horizon_minutes=horizon_minutes,
            features_used=self.feature_columns
        )
        
        logger.info(f"Generated predictions for {intersection_id} with confidence {confidence:.3f}")
        return result
    
    def _calculate_prediction_confidence(self, recent_states: List[TrafficState]) -> float:
        """
        Calculate prediction confidence based on recent data quality and patterns.
        
        Args:
            recent_states: Recent traffic states
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not recent_states:
            return 0.0
        
        # Base confidence from data quality
        avg_confidence = np.mean([state.prediction_confidence for state in recent_states])
        
        # Adjust based on data consistency
        vehicle_counts = [state.get_total_vehicles() for state in recent_states]
        if len(vehicle_counts) > 1:
            cv = np.std(vehicle_counts) / (np.mean(vehicle_counts) + 1e-6)  # Coefficient of variation
            consistency_factor = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
        else:
            consistency_factor = 1.0
        
        # Combine factors
        confidence = (avg_confidence * 0.7 + consistency_factor * 0.3)
        
        return min(1.0, max(0.0, confidence))
    
    def get_prediction_confidence(self, recent_states: List[TrafficState]) -> float:
        """
        Get the current prediction confidence level.
        
        Args:
            recent_states: Recent traffic states for confidence calculation
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        return self._calculate_prediction_confidence(recent_states)
    
    def is_high_confidence(self, confidence: float) -> bool:
        """Check if confidence level is high enough for reliable predictions."""
        return confidence >= self.high_confidence_threshold
    
    def is_low_confidence(self, confidence: float) -> bool:
        """Check if confidence level is too low for reliable predictions."""
        return confidence < self.low_confidence_threshold
    
    def save_model(self) -> None:
        """Save the trained model and scaler to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Save model state and metadata
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'training_history': self.training_history
        }
        
        torch.save(save_data, self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")
    
    def load_model(self) -> bool:
        """
        Load a previously trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_save_path):
            logger.warning(f"Model file not found: {self.model_save_path}")
            return False
        
        try:
            save_data = torch.load(self.model_save_path)
            
            # Restore configuration
            self.feature_columns = save_data['feature_columns']
            self.sequence_length = save_data['sequence_length']
            self.prediction_horizon = save_data['prediction_horizon']
            self.scaler = save_data['scaler']
            self.training_history = save_data.get('training_history', [])
            
            # Recreate and load model
            input_size = len(self.feature_columns)
            self.model = TrafficLSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                output_size=self.prediction_horizon
            )
            self.model.load_state_dict(save_data['model_state_dict'])
            self.model.eval()
            
            self.is_trained = True
            logger.info(f"Model loaded successfully from {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_fallback_prediction(self, intersection_id: str, 
                              recent_states: List[TrafficState],
                              horizon_minutes: int = 30) -> PredictionResult:
        """
        Generate fallback predictions using simple statistical methods.
        
        Args:
            intersection_id: ID of the intersection
            recent_states: Recent traffic states
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            PredictionResult with simple statistical predictions
        """
        logger.warning("Using fallback prediction method due to low model confidence")
        
        if not recent_states:
            # Return zero predictions if no data
            num_predictions = max(1, horizon_minutes // 5)  # 5-minute intervals
            return PredictionResult(
                intersection_id=intersection_id,
                predictions=[0.0] * num_predictions,
                timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(1, num_predictions+1)],
                confidence=0.0,
                horizon_minutes=horizon_minutes,
                features_used=['fallback']
            )
        
        # Use moving average of recent vehicle counts
        recent_counts = [state.get_total_vehicles() for state in recent_states[-6:]]  # Last 30 minutes
        avg_count = np.mean(recent_counts)
        
        # Add some variation based on time of day
        current_hour = recent_states[-1].timestamp.hour
        if current_hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
            multiplier = 1.2
        elif current_hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night hours
            multiplier = 0.5
        else:
            multiplier = 1.0
        
        predicted_count = avg_count * multiplier
        
        # Generate predictions (assume constant for simplicity)
        num_predictions = max(1, horizon_minutes // 5)
        predictions = [predicted_count] * num_predictions
        
        # Generate timestamps
        last_timestamp = recent_states[-1].timestamp
        timestamps = [
            last_timestamp + timedelta(minutes=(i+1)*5) 
            for i in range(num_predictions)
        ]
        
        return PredictionResult(
            intersection_id=intersection_id,
            predictions=predictions,
            timestamps=timestamps,
            confidence=0.3,  # Low confidence for fallback
            horizon_minutes=horizon_minutes,
            features_used=['moving_average', 'time_of_day']
        )