#!/usr/bin/env python3
"""
Basic usage example for Smart Traffic Management System core components.
This demonstrates the data models and configuration management.
"""

import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import TrafficState, VehicleDetection, SignalAction
from config import ConfigManager, IntersectionConfig
from utils import setup_logging

def main():
    """Demonstrate basic usage of core components."""
    
    # Set up logging
    setup_logging(log_level="INFO")
    print("Smart Traffic Management System - Basic Usage Example")
    print("=" * 60)
    
    # 1. Create and validate data models
    print("\n1. Creating Traffic Data Models:")
    
    # Create a traffic state
    traffic_state = TrafficState(
        intersection_id="main_oak_001",
        timestamp=datetime.now(),
        vehicle_counts={"north": 8, "south": 5, "east": 3, "west": 6},
        queue_lengths={"north": 45.2, "south": 28.1, "east": 15.5, "west": 32.8},
        wait_times={"north": 52.3, "south": 35.7, "east": 28.1, "west": 41.2},
        signal_phase="north_south_green",
        prediction_confidence=0.87
    )
    
    print(f"   Traffic State: {traffic_state.intersection_id}")
    print(f"   Total Vehicles: {traffic_state.get_total_vehicles()}")
    print(f"   Average Wait Time: {traffic_state.get_average_wait_time():.1f}s")
    
    # Create vehicle detections
    detections = [
        VehicleDetection(
            vehicle_id="car_001",
            vehicle_type="car",
            position=(125.3, 87.6),
            lane="north_lane_1",
            confidence=0.94,
            timestamp=datetime.now()
        ),
        VehicleDetection(
            vehicle_id="truck_002",
            vehicle_type="truck",
            position=(98.7, 156.2),
            lane="south_lane_2",
            confidence=0.89,
            timestamp=datetime.now()
        )
    ]
    
    print(f"   Vehicle Detections: {len(detections)} vehicles detected")
    for detection in detections:
        print(f"     - {detection.vehicle_type} at {detection.position} (confidence: {detection.confidence:.2f})")
    
    # Create signal action
    signal_action = SignalAction(
        intersection_id="main_oak_001",
        phase_adjustments={"north_south_green": 15, "east_west_green": -8},
        priority_direction="north",
        reasoning="Heavy northbound traffic detected, extending green phase"
    )
    
    print(f"   Signal Action: {signal_action.reasoning}")
    print(f"   Total Adjustment: {signal_action.get_total_adjustment()}s")
    
    # 2. Configuration Management
    print("\n2. Configuration Management:")
    
    # Create config manager
    config_manager = ConfigManager("examples/example_config.json")
    
    # Get system configuration
    sys_config = config_manager.get_system_config()
    print(f"   Camera FPS: {sys_config.camera_fps}")
    print(f"   Detection Threshold: {sys_config.detection_confidence_threshold}")
    print(f"   Learning Rate: {sys_config.learning_rate}")
    
    # Create intersection configuration
    intersection_config = IntersectionConfig("main_oak_001", "Main St & Oak Ave")
    intersection_config.add_lane("north", 3, 120.0)
    intersection_config.add_lane("south", 3, 120.0)
    intersection_config.add_lane("east", 2, 100.0)
    intersection_config.add_lane("west", 2, 100.0)
    
    intersection_config.add_camera_position("north", 0.0, 60.0, 180.0)
    intersection_config.add_camera_position("south", 0.0, -60.0, 0.0)
    intersection_config.add_camera_position("east", 60.0, 0.0, 270.0)
    intersection_config.add_camera_position("west", -60.0, 0.0, 90.0)
    
    intersection_config.set_signal_phases(["north_south_green", "east_west_green"])
    intersection_config.set_default_phase_timings({
        "north_south_green": 50,
        "east_west_green": 45,
        "yellow": 3,
        "all_red": 2
    })
    
    # Validate and add to config manager
    intersection_config.validate()
    config_manager.add_intersection_config("main_oak_001", intersection_config.to_dict())
    
    print(f"   Intersection: {intersection_config.name}")
    print(f"   Total Lanes: {intersection_config.get_total_lanes()}")
    print(f"   Directions: {', '.join(intersection_config.get_directions())}")
    
    # Save configuration
    config_manager.save_config()
    print("   Configuration saved successfully!")
    
    print("\n3. System Ready!")
    print("   All core components initialized and validated.")
    print("   Ready for traffic simulation and optimization.")
    print("\nNext steps:")
    print("   - Implement camera feed processing (Task 2)")
    print("   - Develop RL agent for signal optimization (Task 3)")
    print("   - Create traffic prediction engine (Task 4)")
    print("   - Build Streamlit dashboard (Task 5)")

if __name__ == "__main__":
    main()