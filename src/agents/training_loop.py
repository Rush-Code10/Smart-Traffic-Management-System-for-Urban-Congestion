"""Training loop for Q-learning agent with traffic scenarios."""

import logging
import time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from ..models.traffic_state import TrafficState
from ..models.signal_action import SignalAction
from ..config.intersection_config import IntersectionConfig
from .q_learning_agent import QLearningAgent, QLearningConfig
from .signal_control_manager import SignalControlManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    max_episodes: int = 1000
    episode_length: int = 300  # seconds
    time_step: int = 5  # seconds between decisions
    convergence_threshold: float = 0.01  # reward change threshold for convergence
    convergence_window: int = 50  # episodes to check for convergence
    save_interval: int = 100  # episodes between model saves
    log_interval: int = 10  # episodes between progress logs
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if not (0.0 < self.convergence_threshold < 1.0):
            raise ValueError("convergence_threshold must be between 0 and 1")
        if self.convergence_window <= 0:
            raise ValueError("convergence_window must be positive")


class TrafficScenarioGenerator:
    """Generates realistic traffic scenarios for training."""
    
    def __init__(self, intersection_config: IntersectionConfig):
        """Initialize scenario generator.
        
        Args:
            intersection_config: Configuration for the intersection
        """
        self.intersection_config = intersection_config
        self.directions = intersection_config.get_directions()
        
        # Traffic patterns for different scenarios
        self.scenarios = {
            'light': {'base_rate': 0.3, 'variation': 0.1},
            'moderate': {'base_rate': 0.6, 'variation': 0.2},
            'heavy': {'base_rate': 0.9, 'variation': 0.3},
            'rush_hour': {'base_rate': 1.2, 'variation': 0.4},
            'asymmetric': {'base_rate': 0.7, 'variation': 0.5}  # Uneven traffic distribution
        }
    
    def generate_traffic_sequence(self, scenario: str, duration: int, time_step: int) -> List[TrafficState]:
        """Generate a sequence of traffic states for training.
        
        Args:
            scenario: Traffic scenario name
            duration: Duration in seconds
            time_step: Time step between states in seconds
            
        Returns:
            List of traffic states
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self.scenarios.keys())}")
        
        scenario_config = self.scenarios[scenario]
        base_rate = scenario_config['base_rate']
        variation = scenario_config['variation']
        
        states = []
        start_time = datetime.now()
        
        for step in range(0, duration, time_step):
            timestamp = start_time + timedelta(seconds=step)
            
            # Generate traffic for each direction
            vehicle_counts = {}
            queue_lengths = {}
            wait_times = {}
            
            for direction in self.directions:
                # Add some randomness and time-based variation
                time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * step / duration)  # Cyclical variation
                direction_factor = self._get_direction_factor(direction, scenario)
                
                rate = base_rate * time_factor * direction_factor
                noise = np.random.normal(0, variation)
                final_rate = max(0, rate + noise)
                
                # Generate realistic values
                max_vehicles = int(self.intersection_config.get_lane_count(direction) * 10)
                vehicle_count = int(final_rate * max_vehicles)
                
                # Queue length correlates with vehicle count
                max_queue = self.intersection_config.get_lane_length(direction)
                queue_length = min(max_queue, vehicle_count * 5 + np.random.normal(0, 5))
                queue_length = max(0, queue_length)
                
                # Wait time increases with congestion
                base_wait = 30 + (vehicle_count / max_vehicles) * 120
                wait_time = max(0, base_wait + np.random.normal(0, 15))
                
                vehicle_counts[direction] = vehicle_count
                queue_lengths[direction] = queue_length
                wait_times[direction] = wait_time
            
            # Determine current signal phase (simplified)
            phase_duration = 60  # seconds per phase
            phase_index = (step // phase_duration) % len(self.intersection_config.signal_phases)
            current_phase = self.intersection_config.signal_phases[phase_index]
            
            # Ensure prediction confidence stays within valid range
            confidence = 0.8 + np.random.normal(0, 0.1)
            confidence = max(0.0, min(1.0, confidence))
            
            state = TrafficState(
                intersection_id=self.intersection_config.intersection_id,
                timestamp=timestamp,
                vehicle_counts=vehicle_counts,
                queue_lengths=queue_lengths,
                wait_times=wait_times,
                signal_phase=current_phase,
                prediction_confidence=confidence
            )
            
            states.append(state)
        
        logger.debug(f"Generated {len(states)} traffic states for scenario '{scenario}'")
        return states
    
    def _get_direction_factor(self, direction: str, scenario: str) -> float:
        """Get traffic factor for a specific direction and scenario."""
        if scenario == 'asymmetric':
            # Create uneven traffic distribution
            factors = {'north': 1.5, 'south': 0.5, 'east': 1.2, 'west': 0.8}
            return factors.get(direction, 1.0)
        
        # Default: even distribution
        return 1.0


class TrainingLoop:
    """Training loop for Q-learning agent."""
    
    def __init__(self, 
                 intersection_config: IntersectionConfig,
                 agent_config: Optional[QLearningConfig] = None,
                 training_config: Optional[TrainingConfig] = None):
        """Initialize training loop.
        
        Args:
            intersection_config: Configuration for the intersection
            agent_config: Q-learning agent configuration
            training_config: Training loop configuration
        """
        self.intersection_config = intersection_config
        self.training_config = training_config or TrainingConfig()
        self.training_config.validate()
        
        # Initialize components
        self.agent = QLearningAgent(
            intersection_id=intersection_config.intersection_id,
            signal_phases=intersection_config.signal_phases,
            config=agent_config
        )
        
        self.signal_manager = SignalControlManager(intersection_config)
        self.scenario_generator = TrafficScenarioGenerator(intersection_config)
        
        # Training state
        self.current_episode = 0
        self.training_start_time: Optional[datetime] = None
        self.episode_rewards: List[float] = []
        self.convergence_achieved = False
        
        # Callbacks
        self.episode_callback: Optional[Callable[[int, float, Dict], None]] = None
        self.convergence_callback: Optional[Callable[[int], None]] = None
        
        logger.info(f"Training loop initialized for intersection {intersection_config.intersection_id}")
    
    def set_episode_callback(self, callback: Callable[[int, float, Dict], None]) -> None:
        """Set callback function called after each episode.
        
        Args:
            callback: Function(episode_num, reward, stats) called after each episode
        """
        self.episode_callback = callback
    
    def set_convergence_callback(self, callback: Callable[[int], None]) -> None:
        """Set callback function called when convergence is achieved.
        
        Args:
            callback: Function(episode_num) called when training converges
        """
        self.convergence_callback = callback
    
    def train(self, scenarios: Optional[List[str]] = None, model_save_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the training loop.
        
        Args:
            scenarios: List of traffic scenarios to train on. If None, uses all scenarios.
            model_save_path: Path to save the trained model. If None, model is not saved.
            
        Returns:
            Training results and statistics
        """
        if scenarios is None:
            scenarios = list(self.scenario_generator.scenarios.keys())
        
        self.training_start_time = datetime.now()
        logger.info(f"Starting training with scenarios: {scenarios}")
        logger.info(f"Training config: {self.training_config}")
        
        try:
            for episode in range(self.training_config.max_episodes):
                self.current_episode = episode + 1
                
                # Select random scenario for this episode
                scenario = np.random.choice(scenarios)
                
                # Run episode
                episode_reward = self._run_episode(scenario)
                self.episode_rewards.append(episode_reward)
                
                # Check for convergence
                if self._check_convergence():
                    logger.info(f"Convergence achieved at episode {self.current_episode}")
                    self.convergence_achieved = True
                    if self.convergence_callback:
                        self.convergence_callback(self.current_episode)
                    break
                
                # Periodic logging
                if self.current_episode % self.training_config.log_interval == 0:
                    self._log_progress()
                
                # Periodic model saving
                if (model_save_path and 
                    self.current_episode % self.training_config.save_interval == 0):
                    save_path = f"{model_save_path}_episode_{self.current_episode}.pkl"
                    self.agent.save_model(save_path)
                
                # Episode callback
                if self.episode_callback:
                    stats = self._get_training_stats()
                    self.episode_callback(self.current_episode, episode_reward, stats)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Final model save
            if model_save_path:
                final_path = f"{model_save_path}_final.pkl"
                self.agent.save_model(final_path)
                logger.info(f"Final model saved to {final_path}")
        
        return self._get_training_results()
    
    def _run_episode(self, scenario: str) -> float:
        """Run a single training episode.
        
        Args:
            scenario: Traffic scenario to use
            
        Returns:
            Total reward for the episode
        """
        # Generate traffic sequence
        traffic_states = self.scenario_generator.generate_traffic_sequence(
            scenario=scenario,
            duration=self.training_config.episode_length,
            time_step=self.training_config.time_step
        )
        
        # Reset signal manager for new episode
        self.signal_manager.reset_to_defaults()
        
        # Train the agent on this episode
        episode_reward = self.agent.train_episode(traffic_states)
        
        return episode_reward
    
    def _check_convergence(self) -> bool:
        """Check if training has converged."""
        if len(self.episode_rewards) < self.training_config.convergence_window:
            return False
        
        # Check if reward has stabilized
        recent_rewards = self.episode_rewards[-self.training_config.convergence_window:]
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)
        
        # Convergence if standard deviation is small relative to mean
        if reward_mean != 0:
            coefficient_of_variation = reward_std / abs(reward_mean)
            return coefficient_of_variation < self.training_config.convergence_threshold
        
        return reward_std < self.training_config.convergence_threshold
    
    def _log_progress(self) -> None:
        """Log training progress."""
        stats = self._get_training_stats()
        
        logger.info(f"Episode {self.current_episode}/{self.training_config.max_episodes}")
        logger.info(f"Average reward (last 10): {stats['avg_reward_recent']:.2f}")
        logger.info(f"Average reward (all): {stats['avg_reward_total']:.2f}")
        logger.info(f"Epsilon: {stats['epsilon']:.3f}")
        logger.info(f"Q-table size: {stats['q_table_size']}")
        
        if self.training_start_time:
            elapsed = (datetime.now() - self.training_start_time).total_seconds()
            logger.info(f"Training time: {elapsed:.1f}s")
    
    def _get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        agent_stats = self.agent.get_training_stats()
        
        # Calculate additional stats
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        avg_reward_recent = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_reward_total = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        return {
            'episode': self.current_episode,
            'total_episodes': len(self.episode_rewards),
            'avg_reward_recent': avg_reward_recent,
            'avg_reward_total': avg_reward_total,
            'epsilon': agent_stats['epsilon'],
            'q_table_size': agent_stats['q_table_size'],
            'convergence_achieved': self.convergence_achieved,
            'training_time': (datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 0
        }
    
    def _get_training_results(self) -> Dict[str, Any]:
        """Get final training results."""
        stats = self._get_training_stats()
        
        return {
            'success': True,
            'episodes_completed': self.current_episode,
            'convergence_achieved': self.convergence_achieved,
            'final_epsilon': self.agent.config.epsilon,
            'q_table_size': len(self.agent.q_table),
            'total_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'best_episode_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'worst_episode_reward': min(self.episode_rewards) if self.episode_rewards else 0.0,
            'reward_std': np.std(self.episode_rewards) if self.episode_rewards else 0.0,
            'training_time': stats['training_time'],
            'episodes_per_second': self.current_episode / stats['training_time'] if stats['training_time'] > 0 else 0
        }
    
    def evaluate_agent(self, scenario: str, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained agent on a specific scenario.
        
        Args:
            scenario: Traffic scenario to evaluate on
            num_episodes: Number of episodes to run for evaluation
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating agent on scenario '{scenario}' for {num_episodes} episodes")
        
        # Temporarily disable exploration
        original_epsilon = self.agent.config.epsilon
        self.agent.config.epsilon = 0.0
        
        try:
            episode_rewards = []
            
            for episode in range(num_episodes):
                # Generate traffic sequence
                traffic_states = self.scenario_generator.generate_traffic_sequence(
                    scenario=scenario,
                    duration=self.training_config.episode_length,
                    time_step=self.training_config.time_step
                )
                
                # Reset signal manager
                self.signal_manager.reset_to_defaults()
                
                episode_reward = 0.0
                actions_taken = []
                
                # Run episode without learning
                for i in range(len(traffic_states) - 1):
                    current_state = traffic_states[i]
                    next_state = traffic_states[i + 1]
                    
                    # Get action (no exploration)
                    action = self.agent.get_action(current_state)
                    actions_taken.append(action)
                    
                    # Apply action
                    self.signal_manager.apply_signal_action(action)
                    
                    # Calculate reward
                    reward = self.agent._calculate_reward(current_state, next_state, action.phase_adjustments)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            return {
                'scenario': scenario,
                'num_episodes': num_episodes,
                'average_reward': np.mean(episode_rewards),
                'reward_std': np.std(episode_rewards),
                'best_reward': max(episode_rewards),
                'worst_reward': min(episode_rewards),
                'all_rewards': episode_rewards
            }
        
        finally:
            # Restore original epsilon
            self.agent.config.epsilon = original_epsilon