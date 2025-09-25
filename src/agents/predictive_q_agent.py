"""Enhanced Q-learning agent that integrates traffic predictions for proactive signal optimization."""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .q_learning_agent import QLearningAgent, QLearningConfig
from ..models.traffic_state import TrafficState
from ..models.signal_action import SignalAction
from ..processors.prediction_engine import PredictionEngine, PredictionResult
from ..utils.error_handling import handle_error

logger = logging.getLogger(__name__)


@dataclass
class PredictiveConfig(QLearningConfig):
    """Extended configuration for predictive Q-learning agent."""
    prediction_weight: float = 0.3  # Weight for prediction-based rewards
    prediction_horizon: int = 30  # Prediction horizon in minutes
    confidence_threshold: float = 0.6  # Minimum confidence for using predictions
    proactive_adjustment_factor: float = 1.5  # Factor for proactive adjustments
    
    def validate(self) -> None:
        """Validate predictive configuration."""
        super().validate()
        if not (0.0 <= self.prediction_weight <= 1.0):
            raise ValueError("prediction_weight must be between 0 and 1")
        if not isinstance(self.prediction_horizon, int) or self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be a positive integer")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not isinstance(self.proactive_adjustment_factor, (int, float)) or self.proactive_adjustment_factor <= 0:
            raise ValueError("proactive_adjustment_factor must be a positive number")


class PredictiveQLearningAgent(QLearningAgent):
    """Q-learning agent enhanced with traffic prediction capabilities."""
    
    def __init__(self, intersection_id: str, signal_phases: List[str], 
                 prediction_engine: PredictionEngine,
                 config: Optional[PredictiveConfig] = None):
        """
        Initialize predictive Q-learning agent.
        
        Args:
            intersection_id: ID of the intersection this agent controls
            signal_phases: List of signal phases
            prediction_engine: Traffic prediction engine
            config: Predictive Q-learning configuration
        """
        self.predictive_config = config or PredictiveConfig()
        super().__init__(intersection_id, signal_phases, self.predictive_config)
        
        self.prediction_engine = prediction_engine
        self.recent_states: List[TrafficState] = []
        self.prediction_history: List[PredictionResult] = []
        self.proactive_actions_taken = 0
        
        # Enhanced state space that includes prediction information
        self.prediction_bins = [0, 5, 15, 30, 60]  # bins for predicted traffic increase
        
        logger.info(f"Initialized predictive Q-learning agent for intersection {intersection_id}")
        logger.info(f"Prediction horizon: {self.predictive_config.prediction_horizon} minutes")
    
    def update_recent_states(self, traffic_state: TrafficState) -> None:
        """Update the list of recent traffic states for prediction context."""
        self.recent_states.append(traffic_state)
        
        # Keep only the last hour of states (assuming 5-minute intervals)
        max_states = 12
        if len(self.recent_states) > max_states:
            self.recent_states = self.recent_states[-max_states:]
        
        logger.debug(f"Updated recent states, now have {len(self.recent_states)} states")
    
    def get_traffic_prediction(self) -> Optional[PredictionResult]:
        """Get traffic prediction for the current intersection."""
        if len(self.recent_states) < self.prediction_engine.sequence_length:
            logger.debug("Not enough recent states for prediction")
            return None
        
        try:
            prediction = self.prediction_engine.predict_traffic_volume(
                intersection_id=self.intersection_id,
                recent_states=self.recent_states,
                horizon_minutes=self.predictive_config.prediction_horizon
            )
            
            # Store prediction in history
            self.prediction_history.append(prediction)
            
            # Keep only recent predictions
            if len(self.prediction_history) > 20:
                self.prediction_history = self.prediction_history[-20:]
            
            logger.debug(f"Generated prediction with confidence {prediction.confidence:.3f}")
            return prediction
            
        except Exception as e:
            logger.warning(f"Failed to generate prediction: {e}")
            return None
    
    def _discretize_predictive_state(self, traffic_state: TrafficState, 
                                   prediction: Optional[PredictionResult] = None) -> str:
        """Convert traffic state and prediction to discrete state string."""
        # Get base state from parent class
        base_state = self._discretize_state(traffic_state)
        
        if prediction is None or prediction.confidence < self.predictive_config.confidence_threshold:
            # No reliable prediction, use base state
            return f"{base_state}_pred_none"
        
        # Calculate predicted traffic change
        current_vehicles = traffic_state.get_total_vehicles()
        predicted_peak = max(prediction.predictions) if prediction.predictions else current_vehicles
        traffic_increase = predicted_peak - current_vehicles
        
        # Discretize prediction
        pred_bin = self._get_bin_index(traffic_increase, self.prediction_bins)
        confidence_level = "high" if prediction.confidence > 0.8 else "med"
        
        return f"{base_state}_pred_{pred_bin}_{confidence_level}"
    
    def get_action(self, traffic_state: TrafficState) -> SignalAction:
        """Select action using predictive epsilon-greedy policy."""
        # Update recent states
        self.update_recent_states(traffic_state)
        
        # Get prediction
        prediction = self.get_traffic_prediction()
        
        # Use enhanced state representation
        state = self._discretize_predictive_state(traffic_state, prediction)
        
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(len(self.actions))}
        
        # Enhanced action selection considering predictions
        if np.random.random() < self.config.epsilon:
            # Exploration: random action
            action_index = np.random.randint(0, len(self.actions))
            logger.debug(f"Exploration: selected random action {action_index}")
        else:
            # Exploitation with prediction bias
            action_index = self._select_predictive_action(traffic_state, prediction, state)
        
        # Convert action index to SignalAction
        phase_adjustments = self.actions[action_index]
        
        # Check if this is a proactive action
        is_proactive = self._is_proactive_action(traffic_state, prediction, phase_adjustments)
        if is_proactive:
            self.proactive_actions_taken += 1
        
        # Generate enhanced reasoning
        reasoning = self._generate_predictive_reasoning(traffic_state, prediction, phase_adjustments, is_proactive)
        
        return SignalAction(
            intersection_id=self.intersection_id,
            phase_adjustments=phase_adjustments,
            reasoning=reasoning
        )
    
    def _select_predictive_action(self, traffic_state: TrafficState, 
                                prediction: Optional[PredictionResult], 
                                state: str) -> int:
        """Select action considering both Q-values and predictions."""
        q_values = self.q_table[state]
        
        if prediction is None or prediction.confidence < self.predictive_config.confidence_threshold:
            # No reliable prediction, use standard Q-learning
            return max(q_values.keys(), key=lambda k: q_values[k])
        
        # Enhance Q-values with prediction information
        enhanced_values = {}
        current_vehicles = traffic_state.get_total_vehicles()
        predicted_peak = max(prediction.predictions) if prediction.predictions else current_vehicles
        
        for action_idx, q_value in q_values.items():
            action = self.actions[action_idx]
            
            # Calculate prediction bonus
            prediction_bonus = self._calculate_prediction_bonus(
                current_vehicles, predicted_peak, action, prediction.confidence
            )
            
            enhanced_values[action_idx] = q_value + prediction_bonus
        
        best_action = max(enhanced_values.keys(), key=lambda k: enhanced_values[k])
        
        logger.debug(f"Selected action {best_action} with enhanced Q-value {enhanced_values[best_action]:.3f}")
        return best_action
    
    def _calculate_prediction_bonus(self, current_vehicles: int, predicted_peak: float,
                                  action: Dict[str, int], confidence: float) -> float:
        """Calculate bonus for actions that proactively handle predicted traffic."""
        if predicted_peak <= current_vehicles:
            return 0.0  # No increase predicted
        
        traffic_increase = predicted_peak - current_vehicles
        
        # Bonus for extending green time when traffic increase is predicted
        total_extension = sum(max(0, adj) for adj in action.values())
        proactive_bonus = total_extension * traffic_increase * 0.01 * confidence
        
        # Scale by prediction weight
        return proactive_bonus * self.predictive_config.prediction_weight
    
    def _is_proactive_action(self, traffic_state: TrafficState, 
                           prediction: Optional[PredictionResult],
                           adjustments: Dict[str, int]) -> bool:
        """Check if the action is proactive based on predictions."""
        if prediction is None or prediction.confidence < self.predictive_config.confidence_threshold:
            return False
        
        current_vehicles = traffic_state.get_total_vehicles()
        predicted_peak = max(prediction.predictions) if prediction.predictions else current_vehicles
        
        # Consider it proactive if we're extending phases when traffic increase is predicted
        traffic_increase = predicted_peak - current_vehicles
        total_extension = sum(max(0, adj) for adj in adjustments.values())
        
        return traffic_increase > 5 and total_extension > 0
    
    def _generate_predictive_reasoning(self, traffic_state: TrafficState,
                                     prediction: Optional[PredictionResult],
                                     adjustments: Dict[str, int],
                                     is_proactive: bool) -> str:
        """Generate reasoning that includes prediction information."""
        base_reasoning = self._generate_reasoning(traffic_state, adjustments)
        
        if prediction is None or prediction.confidence < self.predictive_config.confidence_threshold:
            return f"{base_reasoning} (no reliable prediction)"
        
        current_vehicles = traffic_state.get_total_vehicles()
        predicted_peak = max(prediction.predictions) if prediction.predictions else current_vehicles
        
        if is_proactive:
            return (f"{base_reasoning} PROACTIVE: Predicted traffic increase from "
                   f"{current_vehicles} to {predicted_peak:.0f} vehicles "
                   f"(confidence: {prediction.confidence:.2f})")
        else:
            return (f"{base_reasoning} Prediction: {predicted_peak:.0f} peak vehicles "
                   f"(confidence: {prediction.confidence:.2f})")
    
    def _calculate_predictive_reward(self, prev_state: TrafficState, current_state: TrafficState,
                                   action: Dict[str, int], prediction: Optional[PredictionResult]) -> float:
        """Calculate reward enhanced with prediction accuracy."""
        # Get base reward
        base_reward = self._calculate_reward(prev_state, current_state, action)
        
        if prediction is None or prediction.confidence < self.predictive_config.confidence_threshold:
            return base_reward
        
        # Add prediction-based reward
        prediction_reward = 0.0
        
        # Reward for proactive actions that prevent congestion
        current_vehicles = current_state.get_total_vehicles()
        prev_vehicles = prev_state.get_total_vehicles()
        
        if len(prediction.predictions) > 0:
            predicted_vehicles = prediction.predictions[0]  # Next time step prediction
            
            # If we predicted increase and took proactive action, reward if it helped
            if predicted_vehicles > prev_vehicles:
                total_extension = sum(max(0, adj) for adj in action.values())
                if total_extension > 0 and current_state.get_average_wait_time() < prev_state.get_average_wait_time():
                    prediction_reward = 5.0 * prediction.confidence
                    logger.debug(f"Proactive action reward: {prediction_reward:.2f}")
        
        total_reward = base_reward + prediction_reward * self.predictive_config.prediction_weight
        return total_reward
    
    def update_q_values(self, prev_state: TrafficState, action: SignalAction,
                       reward: float, current_state: TrafficState,
                       prediction: Optional[PredictionResult] = None) -> None:
        """Update Q-values with prediction-enhanced rewards."""
        # Calculate enhanced reward
        enhanced_reward = self._calculate_predictive_reward(
            prev_state, current_state, action.phase_adjustments, prediction
        )
        
        # Use enhanced state representation
        prev_state_str = self._discretize_predictive_state(prev_state, prediction)
        current_state_str = self._discretize_predictive_state(current_state, prediction)
        
        # Find action index
        action_index = None
        for i, action_dict in enumerate(self.actions):
            if action_dict == action.phase_adjustments:
                action_index = i
                break
        
        if action_index is None:
            logger.error(f"Action not found in action space: {action.phase_adjustments}")
            return
        
        # Initialize Q-values for new states
        if prev_state_str not in self.q_table:
            self.q_table[prev_state_str] = {i: 0.0 for i in range(len(self.actions))}
        if current_state_str not in self.q_table:
            self.q_table[current_state_str] = {i: 0.0 for i in range(len(self.actions))}
        
        # Q-learning update with enhanced reward
        current_q = self.q_table[prev_state_str][action_index]
        max_next_q = max(self.q_table[current_state_str].values())
        
        new_q = current_q + self.config.learning_rate * (
            enhanced_reward + self.config.discount_factor * max_next_q - current_q
        )
        
        self.q_table[prev_state_str][action_index] = new_q
        
        logger.debug(f"Predictive Q-value update: state={prev_state_str}, action={action_index}, "
                    f"old_q={current_q:.3f}, new_q={new_q:.3f}, enhanced_reward={enhanced_reward:.3f}")
    
    def train_episode_with_predictions(self, traffic_states: List[TrafficState]) -> float:
        """Train the agent with prediction-enhanced learning."""
        if len(traffic_states) < 2:
            logger.warning("Need at least 2 traffic states for training")
            return 0.0
        
        episode_reward = 0.0
        
        for i in range(len(traffic_states) - 1):
            prev_state = traffic_states[i]
            current_state = traffic_states[i + 1]
            
            # Update recent states for prediction context
            self.update_recent_states(prev_state)
            
            # Get prediction
            prediction = self.get_traffic_prediction()
            
            # Get action for previous state
            action = self.get_action(prev_state)
            
            # Calculate enhanced reward
            reward = self._calculate_predictive_reward(
                prev_state, current_state, action.phase_adjustments, prediction
            )
            episode_reward += reward
            
            # Update Q-values with prediction context
            self.update_q_values(prev_state, action, reward, current_state, prediction)
        
        # Update exploration rate
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay
        )
        
        # Update training statistics
        self.training_episodes += 1
        self.total_reward += episode_reward
        self.episode_rewards.append(episode_reward)
        
        # Keep only last 100 episode rewards
        if len(self.episode_rewards) > 100:
            self.episode_rewards = self.episode_rewards[-100:]
        
        logger.info(f"Predictive episode {self.training_episodes} completed. "
                   f"Reward: {episode_reward:.2f}, Proactive actions: {self.proactive_actions_taken}")
        
        return episode_reward
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about prediction usage and accuracy."""
        if not self.prediction_history:
            return {'predictions_made': 0}
        
        confidences = [p.confidence for p in self.prediction_history]
        
        return {
            'predictions_made': len(self.prediction_history),
            'average_confidence': np.mean(confidences),
            'high_confidence_predictions': sum(1 for c in confidences if c > 0.8),
            'proactive_actions_taken': self.proactive_actions_taken,
            'recent_confidences': confidences[-5:] if confidences else []
        }
    
    def get_enhanced_training_stats(self) -> Dict[str, Any]:
        """Get enhanced training statistics including prediction metrics."""
        base_stats = self.get_training_stats()
        prediction_stats = self.get_prediction_stats()
        
        return {**base_stats, **prediction_stats}
    
    def use_fallback_when_low_confidence(self, traffic_state: TrafficState) -> SignalAction:
        """Use fallback decision making when prediction confidence is low."""
        logger.info("Using fallback decision making due to low prediction confidence")
        
        # Use the base Q-learning agent behavior
        return super().get_action(traffic_state)