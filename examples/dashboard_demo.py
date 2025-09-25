"""Demo script for the Smart Traffic Management System Dashboard."""

import sys
import os
from pathlib import Path
import time
import logging
from datetime import datetime, timedelta

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_demo_environment():
    """Set up the demo environment with sample data."""
    try:
        from config.config_manager import ConfigManager
        from models.traffic_state import TrafficState
        from agents.signal_control_manager import SignalControlManager
        from processors.traffic_aggregator import TrafficAggregator
        from agents.q_learning_agent import QLearningAgent
        from processors.prediction_engine import PredictionEngine
        from config.intersection_config import IntersectionConfig
        
        logger.info("Setting up demo environment...")
        
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Create demo intersection if it doesn't exist
        demo_intersection_config = {
            'name': 'Demo Intersection - Main St & Broadway',
            'geometry': {
                'lanes': {
                    'north': {'count': 3, 'length': 120},
                    'south': {'count': 3, 'length': 120},
                    'east': {'count': 2, 'length': 100},
                    'west': {'count': 2, 'length': 100}
                },
                'signal_phases': ['north_south_green', 'east_west_green'],
                'default_phase_timings': {
                    'north_south_green': 50,
                    'east_west_green': 40,
                    'yellow': 3,
                    'all_red': 2
                }
            },
            'camera_positions': {
                'north': {'x': 0, 'y': 60, 'angle': 180},
                'south': {'x': 0, 'y': -60, 'angle': 0},
                'east': {'x': 50, 'y': 0, 'angle': 270},
                'west': {'x': -50, 'y': 0, 'angle': 90}
            }
        }
        
        config_manager.add_intersection_config('demo_intersection', demo_intersection_config)
        config_manager.save_config()
        
        logger.info("Demo intersection configuration created")
        
        # Initialize system components for demo
        intersection_config = IntersectionConfig(
            intersection_id='demo_intersection',
            **demo_intersection_config
        )
        
        # Create signal control manager
        signal_manager = SignalControlManager(intersection_config)
        
        # Create traffic aggregator
        traffic_aggregator = TrafficAggregator('demo_intersection', demo_intersection_config)
        
        # Create RL agent
        rl_agent = QLearningAgent('demo_intersection', demo_intersection_config['geometry']['signal_phases'])
        
        # Create prediction engine
        prediction_engine = PredictionEngine()
        
        logger.info("Demo system components initialized")
        
        return {
            'config_manager': config_manager,
            'signal_manager': signal_manager,
            'traffic_aggregator': traffic_aggregator,
            'rl_agent': rl_agent,
            'prediction_engine': prediction_engine
        }
        
    except Exception as e:
        logger.error(f"Error setting up demo environment: {e}")
        return None

def generate_demo_traffic_data(components, duration_minutes=60):
    """Generate realistic demo traffic data."""
    logger.info(f"Generating {duration_minutes} minutes of demo traffic data...")
    
    traffic_states = []
    current_time = datetime.now() - timedelta(minutes=duration_minutes)
    
    for minute in range(duration_minutes):
        # Simulate realistic traffic patterns based on time
        hour = (current_time + timedelta(minutes=minute)).hour
        
        # Base traffic levels
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_traffic = 25
            congestion_factor = 1.5
        elif 10 <= hour <= 16:  # Daytime
            base_traffic = 15
            congestion_factor = 1.0
        elif 20 <= hour <= 22:  # Evening
            base_traffic = 12
            congestion_factor = 0.8
        else:  # Night/early morning
            base_traffic = 5
            congestion_factor = 0.5
        
        # Add some randomness and patterns
        import numpy as np
        
        # Vehicle counts with directional bias
        vehicle_counts = {
            'north': max(0, int(base_traffic * 1.2 + np.random.normal(0, 3))),
            'south': max(0, int(base_traffic * 1.1 + np.random.normal(0, 3))),
            'east': max(0, int(base_traffic * 0.9 + np.random.normal(0, 2))),
            'west': max(0, int(base_traffic * 0.8 + np.random.normal(0, 2)))
        }
        
        # Queue lengths based on vehicle counts and congestion
        queue_lengths = {
            direction: max(0, count * 2.5 * congestion_factor + np.random.normal(0, 5))
            for direction, count in vehicle_counts.items()
        }
        
        # Wait times based on queue lengths
        wait_times = {
            direction: max(0, queue_length * 1.8 + np.random.normal(0, 8))
            for direction, queue_length in queue_lengths.items()
        }
        
        # Create traffic state
        traffic_state = TrafficState(
            intersection_id='demo_intersection',
            timestamp=current_time + timedelta(minutes=minute),
            vehicle_counts=vehicle_counts,
            queue_lengths=queue_lengths,
            wait_times=wait_times,
            signal_phase='north_south_green' if minute % 2 == 0 else 'east_west_green',
            prediction_confidence=0.7 + np.random.normal(0, 0.1)
        )
        
        traffic_states.append(traffic_state)
    
    logger.info(f"Generated {len(traffic_states)} traffic state records")
    return traffic_states

def train_demo_models(components, traffic_states):
    """Train the RL agent and prediction engine with demo data."""
    logger.info("Training demo models...")
    
    try:
        # Train RL agent with sample episodes
        rl_agent = components['rl_agent']
        
        # Create training episodes from traffic states
        episode_length = 10
        for i in range(0, len(traffic_states) - episode_length, episode_length):
            episode_states = traffic_states[i:i + episode_length]
            reward = rl_agent.train_episode(episode_states)
            logger.info(f"RL training episode {i//episode_length + 1} completed with reward: {reward:.2f}")
        
        # Train prediction engine
        prediction_engine = components['prediction_engine']
        
        if len(traffic_states) >= 50:  # Need minimum data for training
            logger.info("Training LSTM prediction model...")
            training_history = prediction_engine.train_model(
                traffic_states, 
                epochs=20,  # Reduced for demo
                batch_size=16
            )
            logger.info("Prediction model training completed")
        else:
            logger.warning("Insufficient data for prediction model training")
        
        logger.info("Demo model training completed")
        
    except Exception as e:
        logger.error(f"Error training demo models: {e}")

def simulate_real_time_updates(components, duration_seconds=300):
    """Simulate real-time traffic updates for the dashboard."""
    logger.info(f"Starting real-time simulation for {duration_seconds} seconds...")
    
    import numpy as np
    
    signal_manager = components['signal_manager']
    traffic_aggregator = components['traffic_aggregator']
    rl_agent = components['rl_agent']
    
    start_time = time.time()
    update_count = 0
    
    while time.time() - start_time < duration_seconds:
        try:
            # Generate current traffic state
            current_hour = datetime.now().hour
            
            # Simulate realistic current traffic
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
                base_traffic = 20
            elif 10 <= current_hour <= 16:
                base_traffic = 12
            else:
                base_traffic = 6
            
            vehicle_counts = {
                'north': max(0, int(base_traffic + np.random.normal(0, 4))),
                'south': max(0, int(base_traffic + np.random.normal(0, 4))),
                'east': max(0, int(base_traffic * 0.8 + np.random.normal(0, 3))),
                'west': max(0, int(base_traffic * 0.8 + np.random.normal(0, 3)))
            }
            
            queue_lengths = {
                direction: max(0, count * 2.0 + np.random.normal(0, 3))
                for direction, count in vehicle_counts.items()
            }
            
            wait_times = {
                direction: max(0, queue_length * 1.5 + np.random.normal(0, 5))
                for direction, queue_length in queue_lengths.items()
            }
            
            current_state = TrafficState(
                intersection_id='demo_intersection',
                timestamp=datetime.now(),
                vehicle_counts=vehicle_counts,
                queue_lengths=queue_lengths,
                wait_times=wait_times,
                signal_phase=signal_manager.get_current_signal_state().current_cycle_phase,
                prediction_confidence=0.75 + np.random.normal(0, 0.1)
            )
            
            # Get RL agent action
            action = rl_agent.get_action(current_state)
            
            # Apply action to signal manager
            signal_manager.apply_signal_action(action)
            
            # Update traffic aggregator
            signal_manager.update_signal_phase(current_state.signal_phase)
            
            update_count += 1
            
            if update_count % 10 == 0:
                logger.info(f"Real-time update {update_count}: "
                           f"Total vehicles: {current_state.get_total_vehicles()}, "
                           f"Avg wait: {current_state.get_average_wait_time():.1f}s")
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            logger.info("Real-time simulation interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in real-time simulation: {e}")
            time.sleep(1)
    
    logger.info(f"Real-time simulation completed after {update_count} updates")

def run_dashboard_demo():
    """Run the complete dashboard demo."""
    logger.info("Starting Smart Traffic Management System Dashboard Demo")
    
    # Setup demo environment
    components = setup_demo_environment()
    if not components:
        logger.error("Failed to set up demo environment")
        return
    
    # Generate historical traffic data
    traffic_states = generate_demo_traffic_data(components, duration_minutes=120)
    
    # Train models with demo data
    train_demo_models(components, traffic_states)
    
    logger.info("Demo setup completed successfully!")
    logger.info("=" * 60)
    logger.info("DASHBOARD DEMO READY")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To start the dashboard, run:")
    logger.info("  streamlit run src/dashboard/main_dashboard.py")
    logger.info("")
    logger.info("Or use the launcher script:")
    logger.info("  python src/dashboard/run_dashboard.py")
    logger.info("")
    logger.info("The dashboard will be available at: http://localhost:8501")
    logger.info("")
    logger.info("Demo Features Available:")
    logger.info("- Real-time traffic monitoring for 'Demo Intersection'")
    logger.info("- Performance metrics with before/after comparisons")
    logger.info("- Manual override controls for emergency situations")
    logger.info("- Analytics with traffic trends and predictions")
    logger.info("- Data export functionality")
    logger.info("")
    
    # Ask if user wants to run real-time simulation
    try:
        response = input("Would you like to run real-time traffic simulation? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            logger.info("Starting real-time simulation...")
            logger.info("You can now open the dashboard in another terminal to see live updates")
            simulate_real_time_updates(components, duration_seconds=600)  # 10 minutes
        else:
            logger.info("Skipping real-time simulation")
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    
    logger.info("Dashboard demo completed")

def main():
    """Main function for the demo script."""
    print("Smart Traffic Management System - Dashboard Demo")
    print("=" * 50)
    print()
    
    try:
        run_dashboard_demo()
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\nDemo failed: {e}")
        print("Please check the logs for more details.")
    
    print("\nDemo script finished.")

if __name__ == "__main__":
    main()