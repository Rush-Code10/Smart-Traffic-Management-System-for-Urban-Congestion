"""
Demonstration of LSTM-based traffic prediction engine.

This example shows how to:
1. Generate synthetic traffic data with realistic patterns
2. Train an LSTM model for traffic prediction
3. Make predictions and evaluate confidence
4. Integrate predictions with Q-learning agent
5. Handle fallback scenarios when confidence is low
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.processors.prediction_engine import PredictionEngine
from src.agents.predictive_q_agent import PredictiveQLearningAgent, PredictiveConfig
from src.models.traffic_state import TrafficState
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def generate_realistic_traffic_data(num_days: int = 7, interval_minutes: int = 5) -> list:
    """
    Generate realistic traffic data with daily and weekly patterns.
    
    Args:
        num_days: Number of days to generate data for
        interval_minutes: Interval between data points in minutes
        
    Returns:
        List of TrafficState objects
    """
    logger.info(f"Generating {num_days} days of traffic data with {interval_minutes}-minute intervals")
    
    states = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)  # Start on Monday
    points_per_day = 24 * 60 // interval_minutes
    
    for day in range(num_days):
        is_weekend = day >= 5  # Saturday and Sunday
        
        for point in range(points_per_day):
            timestamp = base_time + timedelta(days=day, minutes=point * interval_minutes)
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Base traffic patterns
            if is_weekend:
                # Weekend patterns - later start, more even distribution
                if 10 <= hour <= 14:  # Weekend shopping hours
                    base_vehicles = np.random.poisson(20)
                elif 18 <= hour <= 21:  # Weekend evening
                    base_vehicles = np.random.poisson(15)
                elif 1 <= hour <= 6:  # Night
                    base_vehicles = np.random.poisson(2)
                else:
                    base_vehicles = np.random.poisson(8)
            else:
                # Weekday patterns
                if 7 <= hour <= 9:  # Morning rush
                    base_vehicles = np.random.poisson(35)
                elif 17 <= hour <= 19:  # Evening rush
                    base_vehicles = np.random.poisson(40)
                elif 11 <= hour <= 14:  # Lunch time
                    base_vehicles = np.random.poisson(20)
                elif 22 <= hour or hour <= 5:  # Night
                    base_vehicles = np.random.poisson(3)
                else:
                    base_vehicles = np.random.poisson(12)
            
            # Add some randomness and ensure non-negative
            vehicles = max(0, base_vehicles + np.random.randint(-3, 4))
            
            # Queue lengths and wait times correlate with vehicle count
            queue_base = vehicles * 1.5 + np.random.exponential(5)
            wait_base = vehicles * 2 + np.random.exponential(10)
            
            # Distribute across directions
            north_vehicles = np.random.binomial(vehicles, 0.4)
            south_vehicles = vehicles - north_vehicles
            
            # Calculate confidence and ensure it's in valid range
            confidence = 0.8 + np.random.normal(0, 0.1)
            confidence = max(0.0, min(1.0, confidence))
            
            # Create traffic state
            state = TrafficState(
                intersection_id="main_intersection",
                timestamp=timestamp,
                vehicle_counts={
                    "north": north_vehicles,
                    "south": south_vehicles
                },
                queue_lengths={
                    "north": queue_base * 0.6,
                    "south": queue_base * 0.4
                },
                wait_times={
                    "north": max(0.0, wait_base + np.random.normal(0, 5)),
                    "south": max(0.0, wait_base + np.random.normal(0, 5))
                },
                signal_phase="north_south" if point % 2 == 0 else "east_west",
                prediction_confidence=confidence
            )
            
            states.append(state)
    
    logger.info(f"Generated {len(states)} traffic states")
    return states


def train_prediction_model(traffic_data: list) -> PredictionEngine:
    """
    Train LSTM prediction model on traffic data.
    
    Args:
        traffic_data: List of TrafficState objects
        
    Returns:
        Trained PredictionEngine
    """
    logger.info("Training LSTM prediction model")
    
    # Initialize prediction engine
    engine = PredictionEngine(
        sequence_length=12,  # Use 1 hour of history (12 * 5 minutes)
        prediction_horizon=6,  # Predict next 30 minutes (6 * 5 minutes)
        model_save_path="models/demo_traffic_lstm.pth"
    )
    
    # Train the model
    history = engine.train_model(
        traffic_data,
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        validation_split=0.2
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Show final losses
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    logger.info(f"Training completed. Final train loss: {final_train_loss:.4f}, "
               f"Final validation loss: {final_val_loss:.4f}")
    
    return engine


def demonstrate_predictions(engine: PredictionEngine, traffic_data: list):
    """
    Demonstrate traffic predictions with different scenarios.
    
    Args:
        engine: Trained prediction engine
        traffic_data: Historical traffic data
    """
    logger.info("Demonstrating traffic predictions")
    
    # Test predictions at different times of day
    test_scenarios = [
        {"name": "Morning Rush Hour", "start_hour": 8},
        {"name": "Lunch Time", "start_hour": 12},
        {"name": "Evening Rush Hour", "start_hour": 18},
        {"name": "Night Time", "start_hour": 23}
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, scenario in enumerate(test_scenarios):
        # Find states around the target hour
        target_states = []
        for state in traffic_data[-200:]:  # Use recent data
            if state.timestamp.hour == scenario["start_hour"]:
                target_states.append(state)
        
        if len(target_states) < 15:
            logger.warning(f"Not enough data for {scenario['name']} scenario")
            continue
        
        # Use the last 15 states for prediction context
        recent_states = target_states[-15:]
        
        # Make prediction
        try:
            result = engine.predict_traffic_volume(
                "main_intersection",
                recent_states,
                horizon_minutes=30
            )
            
            logger.info(f"{scenario['name']}: Predicted traffic with confidence {result.confidence:.3f}")
            
            # Plot results
            plt.subplot(2, 2, i + 1)
            
            # Historical data
            historical_vehicles = [s.get_total_vehicles() for s in recent_states[-6:]]
            historical_times = list(range(-len(historical_vehicles), 0))
            
            # Predictions
            prediction_times = list(range(1, len(result.predictions) + 1))
            
            plt.plot(historical_times, historical_vehicles, 'b-o', label='Historical', linewidth=2)
            plt.plot(prediction_times, result.predictions, 'r--s', label='Predicted', linewidth=2)
            plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='Now')
            
            plt.title(f"{scenario['name']}\nConfidence: {result.confidence:.3f}")
            plt.xlabel('Time Steps (5-min intervals)')
            plt.ylabel('Total Vehicles')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Failed to predict for {scenario['name']}: {e}")
    
    plt.tight_layout()
    plt.savefig('traffic_predictions_demo.png', dpi=300, bbox_inches='tight')
    logger.info("Prediction plots saved to traffic_predictions_demo.png")


def demonstrate_predictive_agent(engine: PredictionEngine, traffic_data: list):
    """
    Demonstrate the predictive Q-learning agent.
    
    Args:
        engine: Trained prediction engine
        traffic_data: Historical traffic data
    """
    logger.info("Demonstrating predictive Q-learning agent")
    
    # Create predictive agent
    config = PredictiveConfig(
        learning_rate=0.1,
        epsilon=0.2,
        prediction_weight=0.4,
        confidence_threshold=0.6
    )
    
    agent = PredictiveQLearningAgent(
        intersection_id="main_intersection",
        signal_phases=["north_south", "east_west"],
        prediction_engine=engine,
        config=config
    )
    
    # Train agent on recent data
    logger.info("Training predictive agent")
    recent_data = traffic_data[-100:]  # Use last 100 states for training
    
    # Train for several episodes
    episode_rewards = []
    for episode in range(10):
        # Use sliding window of states for each episode
        start_idx = episode * 8
        end_idx = start_idx + 20
        if end_idx > len(recent_data):
            break
        
        episode_states = recent_data[start_idx:end_idx]
        reward = agent.train_episode_with_predictions(episode_states)
        episode_rewards.append(reward)
        
        if episode % 3 == 0:
            logger.info(f"Episode {episode}: Reward = {reward:.2f}")
    
    # Get training statistics
    stats = agent.get_enhanced_training_stats()
    logger.info(f"Training completed. Episodes: {stats['episodes']}, "
               f"Proactive actions: {stats['proactive_actions_taken']}")
    
    # Demonstrate action selection with predictions
    logger.info("Demonstrating action selection")
    
    test_state = recent_data[-1]
    action = agent.get_action(test_state)
    
    logger.info(f"Selected action for current traffic state:")
    logger.info(f"  Phase adjustments: {action.phase_adjustments}")
    logger.info(f"  Reasoning: {action.reasoning}")
    
    # Show prediction statistics
    pred_stats = agent.get_prediction_stats()
    logger.info(f"Prediction statistics:")
    logger.info(f"  Predictions made: {pred_stats['predictions_made']}")
    logger.info(f"  Average confidence: {pred_stats.get('average_confidence', 0):.3f}")
    logger.info(f"  Proactive actions taken: {pred_stats['proactive_actions_taken']}")
    
    return agent


def demonstrate_fallback_scenarios(engine: PredictionEngine):
    """
    Demonstrate fallback prediction scenarios.
    
    Args:
        engine: Trained prediction engine
    """
    logger.info("Demonstrating fallback scenarios")
    
    # Scenario 1: No historical data
    logger.info("Scenario 1: No historical data")
    fallback_result = engine.get_fallback_prediction(
        "test_intersection",
        [],
        horizon_minutes=30
    )
    logger.info(f"Fallback prediction with no data: {fallback_result.predictions}")
    logger.info(f"Confidence: {fallback_result.confidence}")
    
    # Scenario 2: Rush hour fallback
    logger.info("Scenario 2: Rush hour fallback")
    rush_hour_state = TrafficState(
        intersection_id="test_intersection",
        timestamp=datetime(2024, 1, 1, 8, 30, 0),  # Rush hour
        vehicle_counts={"north": 20, "south": 18},
        queue_lengths={"north": 35.0, "south": 30.0},
        wait_times={"north": 75.0, "south": 70.0},
        signal_phase="north_south",
        prediction_confidence=0.7
    )
    
    rush_fallback = engine.get_fallback_prediction(
        "test_intersection",
        [rush_hour_state],
        horizon_minutes=15
    )
    logger.info(f"Rush hour fallback prediction: {rush_fallback.predictions}")
    logger.info(f"Confidence: {rush_fallback.confidence}")
    
    # Scenario 3: Night time fallback
    logger.info("Scenario 3: Night time fallback")
    night_state = TrafficState(
        intersection_id="test_intersection",
        timestamp=datetime(2024, 1, 1, 2, 15, 0),  # Night time
        vehicle_counts={"north": 2, "south": 1},
        queue_lengths={"north": 5.0, "south": 3.0},
        wait_times={"north": 15.0, "south": 12.0},
        signal_phase="north_south",
        prediction_confidence=0.6
    )
    
    night_fallback = engine.get_fallback_prediction(
        "test_intersection",
        [night_state],
        horizon_minutes=20
    )
    logger.info(f"Night time fallback prediction: {night_fallback.predictions}")
    logger.info(f"Confidence: {night_fallback.confidence}")


def main():
    """Main demonstration function."""
    logger.info("Starting LSTM Traffic Prediction Demo")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Step 1: Generate realistic traffic data
        logger.info("Step 1: Generating traffic data")
        traffic_data = generate_realistic_traffic_data(num_days=14, interval_minutes=5)
        
        # Step 2: Train prediction model
        logger.info("Step 2: Training prediction model")
        engine = train_prediction_model(traffic_data)
        
        # Step 3: Demonstrate predictions
        logger.info("Step 3: Demonstrating predictions")
        demonstrate_predictions(engine, traffic_data)
        
        # Step 4: Demonstrate predictive agent
        logger.info("Step 4: Demonstrating predictive agent")
        agent = demonstrate_predictive_agent(engine, traffic_data)
        
        # Step 5: Demonstrate fallback scenarios
        logger.info("Step 5: Demonstrating fallback scenarios")
        demonstrate_fallback_scenarios(engine)
        
        # Step 6: Show confidence analysis
        logger.info("Step 6: Analyzing prediction confidence")
        
        # Test confidence with different data quality
        high_quality_states = []
        low_quality_states = []
        
        for state in traffic_data[-20:]:
            # High quality version
            high_state = TrafficState(
                intersection_id=state.intersection_id,
                timestamp=state.timestamp,
                vehicle_counts=state.vehicle_counts,
                queue_lengths=state.queue_lengths,
                wait_times=state.wait_times,
                signal_phase=state.signal_phase,
                prediction_confidence=0.95
            )
            high_quality_states.append(high_state)
            
            # Low quality version
            low_state = TrafficState(
                intersection_id=state.intersection_id,
                timestamp=state.timestamp,
                vehicle_counts=state.vehicle_counts,
                queue_lengths=state.queue_lengths,
                wait_times=state.wait_times,
                signal_phase=state.signal_phase,
                prediction_confidence=0.2
            )
            low_quality_states.append(low_state)
        
        high_confidence = engine.get_prediction_confidence(high_quality_states)
        low_confidence = engine.get_prediction_confidence(low_quality_states)
        
        logger.info(f"High quality data confidence: {high_confidence:.3f}")
        logger.info(f"Low quality data confidence: {low_confidence:.3f}")
        logger.info(f"High confidence threshold: {engine.high_confidence_threshold}")
        logger.info(f"Low confidence threshold: {engine.low_confidence_threshold}")
        
        # Show final statistics
        logger.info("Demo completed successfully!")
        logger.info(f"Model trained on {len(traffic_data)} traffic states")
        logger.info(f"Prediction engine ready for real-time use")
        
        if hasattr(agent, 'get_enhanced_training_stats'):
            final_stats = agent.get_enhanced_training_stats()
            logger.info(f"Agent training statistics: {final_stats}")
        
        plt.show()  # Display all plots
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()