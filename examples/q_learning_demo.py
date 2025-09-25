"""Demo script for Q-learning traffic signal optimization."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime

from src.config.intersection_config import IntersectionConfig
from src.agents import QLearningAgent, QLearningConfig, TrainingLoop, TrainingConfig
from src.agents.signal_control_manager import SignalControlManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_intersection():
    """Create a sample intersection configuration."""
    config = IntersectionConfig("demo_intersection", "Demo 4-Way Intersection")
    
    # Add lanes for each direction
    config.add_lane("north", 2, 100.0)
    config.add_lane("south", 2, 100.0)
    config.add_lane("east", 1, 80.0)
    config.add_lane("west", 1, 80.0)
    
    # Add camera positions
    config.add_camera_position("north", 0, 50, 180)
    config.add_camera_position("south", 0, -50, 0)
    config.add_camera_position("east", 50, 0, 270)
    config.add_camera_position("west", -50, 0, 90)
    
    # Set signal phases and timings
    config.set_signal_phases(["north_south", "east_west"])
    config.set_default_phase_timings({"north_south": 60, "east_west": 45})
    
    config.validate()
    return config


def demo_q_learning_agent():
    """Demonstrate Q-learning agent functionality."""
    logger.info("=== Q-Learning Agent Demo ===")
    
    # Create intersection configuration
    intersection_config = create_sample_intersection()
    logger.info(f"Created intersection: {intersection_config.name}")
    
    # Create Q-learning agent
    agent_config = QLearningConfig(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,  # Higher exploration for demo
        max_adjustment=20
    )
    
    agent = QLearningAgent(
        intersection_id=intersection_config.intersection_id,
        signal_phases=intersection_config.signal_phases,
        config=agent_config
    )
    
    logger.info(f"Created Q-learning agent with {len(agent.actions)} possible actions")
    
    # Create signal control manager
    signal_manager = SignalControlManager(intersection_config)
    logger.info("Created signal control manager")
    
    # Demonstrate getting current signal state
    current_state = signal_manager.get_current_signal_state()
    logger.info(f"Current signal phase: {current_state.current_cycle_phase}")
    logger.info(f"Active phases: {current_state.get_active_phases()}")
    
    # Create sample traffic state
    from src.models.traffic_state import TrafficState
    
    traffic_state = TrafficState(
        intersection_id=intersection_config.intersection_id,
        timestamp=datetime.now(),
        vehicle_counts={"north": 15, "south": 12, "east": 8, "west": 10},
        queue_lengths={"north": 35.0, "south": 28.0, "east": 20.0, "west": 25.0},
        wait_times={"north": 65.0, "south": 55.0, "east": 40.0, "west": 45.0},
        signal_phase="north_south",
        prediction_confidence=0.85
    )
    
    logger.info(f"Sample traffic state - Total vehicles: {traffic_state.get_total_vehicles()}")
    logger.info(f"Average wait time: {traffic_state.get_average_wait_time():.1f}s")
    
    # Get action from agent
    action = agent.get_action(traffic_state)
    logger.info(f"Agent recommended action: {action.phase_adjustments}")
    logger.info(f"Reasoning: {action.reasoning}")
    
    # Apply action to signal manager
    success = signal_manager.apply_signal_action(action)
    logger.info(f"Action applied successfully: {success}")
    
    if success:
        new_timings = signal_manager.get_phase_timings()
        logger.info(f"Updated phase timings: {new_timings}")
    
    # Show training statistics
    stats = agent.get_training_stats()
    logger.info(f"Agent training stats: {stats}")


def demo_training_loop():
    """Demonstrate training loop functionality."""
    logger.info("\n=== Training Loop Demo ===")
    
    # Create intersection configuration
    intersection_config = create_sample_intersection()
    
    # Create training configuration for quick demo
    training_config = TrainingConfig(
        max_episodes=20,  # Small number for demo
        episode_length=120,  # 2 minutes per episode
        time_step=10,  # 10 second decisions
        log_interval=5  # Log every 5 episodes
    )
    
    agent_config = QLearningConfig(
        learning_rate=0.2,
        epsilon=0.5,  # Start with high exploration
        epsilon_decay=0.95
    )
    
    # Create training loop
    training_loop = TrainingLoop(
        intersection_config=intersection_config,
        agent_config=agent_config,
        training_config=training_config
    )
    
    logger.info("Created training loop")
    
    # Set up callbacks
    def episode_callback(episode, reward, stats):
        if episode % 5 == 0:
            logger.info(f"Episode {episode}: Reward={reward:.2f}, Epsilon={stats['epsilon']:.3f}")
    
    def convergence_callback(episode):
        logger.info(f"Training converged at episode {episode}!")
    
    training_loop.set_episode_callback(episode_callback)
    training_loop.set_convergence_callback(convergence_callback)
    
    # Run training
    logger.info("Starting training...")
    results = training_loop.train(scenarios=['moderate', 'heavy'])
    
    logger.info("Training completed!")
    logger.info(f"Episodes completed: {results['episodes_completed']}")
    logger.info(f"Convergence achieved: {results['convergence_achieved']}")
    logger.info(f"Average reward: {results['average_reward']:.2f}")
    logger.info(f"Training time: {results['training_time']:.1f}s")
    
    # Evaluate trained agent
    logger.info("\nEvaluating trained agent...")
    eval_results = training_loop.evaluate_agent('rush_hour', num_episodes=5)
    logger.info(f"Evaluation on rush_hour scenario:")
    logger.info(f"Average reward: {eval_results['average_reward']:.2f}")
    logger.info(f"Best reward: {eval_results['best_reward']:.2f}")
    logger.info(f"Reward std: {eval_results['reward_std']:.2f}")


def demo_manual_override():
    """Demonstrate manual override functionality."""
    logger.info("\n=== Manual Override Demo ===")
    
    # Create intersection and signal manager
    intersection_config = create_sample_intersection()
    signal_manager = SignalControlManager(intersection_config)
    
    # Show initial state
    state = signal_manager.get_current_signal_state()
    logger.info(f"Initial phase: {state.current_cycle_phase}")
    logger.info(f"Manual override: {state.manual_override}")
    
    # Enable manual override
    success = signal_manager.enable_manual_override("operator_demo")
    logger.info(f"Manual override enabled: {success}")
    
    # Set manual phase
    success = signal_manager.set_manual_phase("east_west", 90)
    logger.info(f"Manual phase set: {success}")
    
    # Check updated state
    state = signal_manager.get_current_signal_state()
    logger.info(f"Current phase: {state.current_cycle_phase}")
    logger.info(f"Override operator: {state.override_operator}")
    
    # Disable manual override
    success = signal_manager.disable_manual_override()
    logger.info(f"Manual override disabled: {success}")
    
    # Show final state
    state = signal_manager.get_current_signal_state()
    logger.info(f"Final manual override status: {state.manual_override}")


if __name__ == "__main__":
    try:
        demo_q_learning_agent()
        demo_training_loop()
        demo_manual_override()
        logger.info("\n=== Demo completed successfully! ===")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise