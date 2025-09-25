"""Q-learning reinforcement learning agent for traffic signal optimization."""

import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import random

from ..models.traffic_state import TrafficState
from ..models.signal_action import SignalAction

logger = logging.getLogger(__name__)


@dataclass
class QLearningConfig:
    """Configuration for Q-learning agent."""
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1  # exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    max_adjustment: int = 30  # maximum seconds to adjust signal timing
    
    def validate(self) -> None:
        """Validate Q-learning configuration."""
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be between 0 and 1")
        if not (0.0 <= self.discount_factor <= 1.0):
            raise ValueError("discount_factor must be between 0 and 1")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be between 0 and 1")
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError("epsilon_decay must be between 0 and 1")
        if not (0.0 <= self.epsilon_min <= 1.0):
            raise ValueError("epsilon_min must be between 0 and 1")
        if not isinstance(self.max_adjustment, int) or self.max_adjustment <= 0:
            raise ValueError("max_adjustment must be a positive integer")


class QLearningAgent:
    """Q-learning agent for optimizing traffic signal timings."""
    
    def __init__(self, intersection_id: str, signal_phases: List[str], config: Optional[QLearningConfig] = None):
        """Initialize Q-learning agent.
        
        Args:
            intersection_id: ID of the intersection this agent controls
            signal_phases: List of signal phases (e.g., ['north_south', 'east_west'])
            config: Q-learning configuration parameters
        """
        self.intersection_id = intersection_id
        self.signal_phases = signal_phases
        self.config = config or QLearningConfig()
        self.config.validate()
        
        # Q-table: state -> action -> Q-value
        self.q_table: Dict[str, Dict[int, float]] = {}
        
        # State discretization parameters
        self.vehicle_count_bins = [0, 5, 10, 20, 50]  # bins for vehicle counts
        self.queue_length_bins = [0, 10, 25, 50, 100]  # bins for queue lengths (meters)
        self.wait_time_bins = [0, 30, 60, 120, 300]  # bins for wait times (seconds)
        
        # Action space: adjustment for each phase (-max_adjustment to +max_adjustment)
        self.actions = self._generate_action_space()
        
        # Training statistics
        self.training_episodes = 0
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []
        
        logger.info(f"Initialized Q-learning agent for intersection {intersection_id}")
        logger.info(f"Signal phases: {signal_phases}")
        logger.info(f"Action space size: {len(self.actions)}")
    
    def _generate_action_space(self) -> List[Dict[str, int]]:
        """Generate all possible actions (phase timing adjustments)."""
        actions = []
        
        # Generate combinations of adjustments for each phase
        adjustment_values = [-self.config.max_adjustment, -15, -5, 0, 5, 15, self.config.max_adjustment]
        
        # For simplicity, we'll use a subset of possible combinations
        # Each action adjusts one phase at a time
        for phase in self.signal_phases:
            for adjustment in adjustment_values:
                if adjustment != 0:  # Skip no-change actions for individual phases
                    action = {p: 0 for p in self.signal_phases}
                    action[phase] = adjustment
                    actions.append(action)
        
        # Add a "no change" action
        actions.append({phase: 0 for phase in self.signal_phases})
        
        return actions
    
    def _discretize_state(self, traffic_state: TrafficState) -> str:
        """Convert continuous traffic state to discrete state string."""
        # Get total values across all directions
        total_vehicles = traffic_state.get_total_vehicles()
        total_queue_length = traffic_state.get_total_queue_length()
        avg_wait_time = traffic_state.get_average_wait_time()
        current_phase = traffic_state.signal_phase
        
        # Discretize continuous values
        vehicle_bin = self._get_bin_index(total_vehicles, self.vehicle_count_bins)
        queue_bin = self._get_bin_index(total_queue_length, self.queue_length_bins)
        wait_bin = self._get_bin_index(avg_wait_time, self.wait_time_bins)
        
        # Create state string
        state = f"v{vehicle_bin}_q{queue_bin}_w{wait_bin}_p{current_phase}"
        return state
    
    def _get_bin_index(self, value: float, bins: List[float]) -> int:
        """Get the bin index for a continuous value."""
        for i, bin_edge in enumerate(bins[1:], 1):
            if value <= bin_edge:
                return i - 1
        return len(bins) - 1  # Last bin for values above the highest edge
    
    def _calculate_reward(self, prev_state: TrafficState, current_state: TrafficState, action: Dict[str, int]) -> float:
        """Calculate reward based on traffic state changes."""
        # Primary reward: reduction in total wait time
        prev_wait = prev_state.get_average_wait_time()
        current_wait = current_state.get_average_wait_time()
        wait_time_reward = (prev_wait - current_wait) * 10  # Scale factor
        
        # Secondary reward: reduction in queue length
        prev_queue = prev_state.get_total_queue_length()
        current_queue = current_state.get_total_queue_length()
        queue_reward = (prev_queue - current_queue) * 0.1
        
        # Penalty for large adjustments (encourage stability)
        adjustment_penalty = sum(abs(adj) for adj in action.values()) * 0.1
        
        # Bonus for handling high traffic efficiently
        traffic_bonus = 0
        if current_state.get_total_vehicles() > 20 and current_wait < prev_wait:
            traffic_bonus = 5
        
        total_reward = wait_time_reward + queue_reward - adjustment_penalty + traffic_bonus
        
        logger.debug(f"Reward calculation: wait_time={wait_time_reward:.2f}, "
                    f"queue={queue_reward:.2f}, penalty={adjustment_penalty:.2f}, "
                    f"bonus={traffic_bonus:.2f}, total={total_reward:.2f}")
        
        return total_reward
    
    def get_action(self, traffic_state: TrafficState) -> SignalAction:
        """Select action using epsilon-greedy policy."""
        state = self._discretize_state(traffic_state)
        
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(len(self.actions))}
        
        # Epsilon-greedy action selection
        if random.random() < self.config.epsilon:
            # Exploration: random action
            action_index = random.randint(0, len(self.actions) - 1)
            logger.debug(f"Exploration: selected random action {action_index}")
        else:
            # Exploitation: best known action
            q_values = self.q_table[state]
            action_index = max(q_values.keys(), key=lambda k: q_values[k])
            logger.debug(f"Exploitation: selected best action {action_index} with Q-value {q_values[action_index]:.3f}")
        
        # Convert action index to SignalAction
        phase_adjustments = self.actions[action_index]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(traffic_state, phase_adjustments)
        
        return SignalAction(
            intersection_id=self.intersection_id,
            phase_adjustments=phase_adjustments,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, traffic_state: TrafficState, adjustments: Dict[str, int]) -> str:
        """Generate human-readable reasoning for the action."""
        total_vehicles = traffic_state.get_total_vehicles()
        avg_wait = traffic_state.get_average_wait_time()
        
        if all(adj == 0 for adj in adjustments.values()):
            return f"No adjustment needed. Traffic: {total_vehicles} vehicles, avg wait: {avg_wait:.1f}s"
        
        adjusted_phases = [phase for phase, adj in adjustments.items() if adj != 0]
        if len(adjusted_phases) == 1:
            phase = adjusted_phases[0]
            adj = adjustments[phase]
            direction = "extending" if adj > 0 else "reducing"
            return f"{direction.capitalize()} {phase} phase by {abs(adj)}s. Traffic: {total_vehicles} vehicles, avg wait: {avg_wait:.1f}s"
        
        return f"Adjusting multiple phases. Traffic: {total_vehicles} vehicles, avg wait: {avg_wait:.1f}s"
    
    def update_q_values(self, prev_state: TrafficState, action: SignalAction, 
                       reward: float, current_state: TrafficState) -> None:
        """Update Q-values using Q-learning update rule."""
        prev_state_str = self._discretize_state(prev_state)
        current_state_str = self._discretize_state(current_state)
        
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
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[prev_state_str][action_index]
        max_next_q = max(self.q_table[current_state_str].values())
        
        new_q = current_q + self.config.learning_rate * (
            reward + self.config.discount_factor * max_next_q - current_q
        )
        
        self.q_table[prev_state_str][action_index] = new_q
        
        logger.debug(f"Q-value update: state={prev_state_str}, action={action_index}, "
                    f"old_q={current_q:.3f}, new_q={new_q:.3f}, reward={reward:.3f}")
    
    def train_episode(self, traffic_states: List[TrafficState]) -> float:
        """Train the agent on a sequence of traffic states."""
        if len(traffic_states) < 2:
            logger.warning("Need at least 2 traffic states for training")
            return 0.0
        
        episode_reward = 0.0
        
        for i in range(len(traffic_states) - 1):
            prev_state = traffic_states[i]
            current_state = traffic_states[i + 1]
            
            # Get action for previous state
            action = self.get_action(prev_state)
            
            # Calculate reward
            reward = self._calculate_reward(prev_state, current_state, action.phase_adjustments)
            episode_reward += reward
            
            # Update Q-values
            self.update_q_values(prev_state, action, reward, current_state)
        
        # Update exploration rate
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay
        )
        
        # Update training statistics
        self.training_episodes += 1
        self.total_reward += episode_reward
        self.episode_rewards.append(episode_reward)
        
        # Keep only last 100 episode rewards for memory efficiency
        if len(self.episode_rewards) > 100:
            self.episode_rewards = self.episode_rewards[-100:]
        
        logger.info(f"Episode {self.training_episodes} completed. "
                   f"Reward: {episode_reward:.2f}, Epsilon: {self.config.epsilon:.3f}")
        
        return episode_reward
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return {
            'episodes': self.training_episodes,
            'total_reward': self.total_reward,
            'average_reward': avg_reward,
            'epsilon': self.config.epsilon,
            'q_table_size': len(self.q_table),
            'recent_rewards': self.episode_rewards[-10:] if self.episode_rewards else []
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the Q-table and configuration to file."""
        model_data = {
            'intersection_id': self.intersection_id,
            'signal_phases': self.signal_phases,
            'config': self.config,
            'q_table': self.q_table,
            'training_episodes': self.training_episodes,
            'total_reward': self.total_reward,
            'episode_rewards': self.episode_rewards
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load Q-table and configuration from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.intersection_id = model_data['intersection_id']
        self.signal_phases = model_data['signal_phases']
        self.config = model_data['config']
        self.q_table = model_data['q_table']
        self.training_episodes = model_data['training_episodes']
        self.total_reward = model_data['total_reward']
        self.episode_rewards = model_data['episode_rewards']
        
        # Regenerate action space in case it changed
        self.actions = self._generate_action_space()
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Loaded {len(self.q_table)} states, {self.training_episodes} episodes")