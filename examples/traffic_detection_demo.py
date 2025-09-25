"""
Demo script showing traffic simulation and vehicle detection system integration.

This example demonstrates how to use the traffic simulation, camera processing,
vehicle counting, and traffic aggregation components together.
"""

import sys
import os
from datetime import datetime, timedelta
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_manager import ConfigManager
from src.processors.traffic_simulator import TrafficSimulator
from src.processors.camera_processor import CameraProcessor
from src.processors.vehicle_counter import VehicleCounter
from src.processors.traffic_aggregator import TrafficAggregator


def main():
    """Run traffic detection demo."""
    print("🚦 Smart Traffic Management System - Detection Demo")
    print("=" * 60)
    
    # Initialize configuration
    print("📋 Initializing system configuration...")
    config_manager = ConfigManager()
    intersection_config = config_manager.get_intersection_config('intersection_001')
    
    if not intersection_config:
        print("❌ No intersection configuration found!")
        return
    
    print(f"✅ Loaded configuration for: {intersection_config['name']}")
    
    # Initialize components
    print("\n🔧 Initializing system components...")
    
    # Traffic simulator for generating synthetic data
    simulator = TrafficSimulator(intersection_config)
    print("✅ Traffic simulator initialized")
    
    # Camera processor for vehicle detection
    camera_config = intersection_config.get('camera_positions', {}).get('north', {})
    processor = CameraProcessor(camera_config)
    print("✅ Camera processor initialized")
    
    # Vehicle counter for tracking and counting
    counter = VehicleCounter(intersection_config)
    print("✅ Vehicle counter initialized")
    
    # Traffic aggregator for comprehensive analysis
    aggregator = TrafficAggregator('intersection_001', intersection_config)
    print("✅ Traffic aggregator initialized")
    
    print("\n🚀 Starting traffic detection simulation...")
    print("-" * 60)
    
    # Run simulation for several cycles
    for cycle in range(5):
        print(f"\n📊 Cycle {cycle + 1}/5 - {datetime.now().strftime('%H:%M:%S')}")
        
        # Generate synthetic traffic data
        timestamp = datetime.now()
        detections = simulator.generate_vehicle_detections(timestamp, duration_seconds=5)
        print(f"🚗 Generated {len(detections)} vehicle detections")
        
        # Process detections through camera processor
        # (In real system, this would process actual camera frames)
        synthetic_frame = processor.create_synthetic_frame(num_vehicles=len(detections))
        camera_detections = processor.process_frame(synthetic_frame, timestamp, 'north')
        print(f"📹 Camera processed {len(camera_detections)} detections")
        
        # Combine detections (use simulator detections for demo)
        all_detections = detections + camera_detections
        
        # Count and track vehicles
        counting_result = counter.process_detections(all_detections)
        print(f"🔢 Counted {counting_result['total_vehicles']} vehicles across lanes")
        
        # Aggregate traffic data
        traffic_state = aggregator.aggregate_traffic_data(
            all_detections, 
            counting_result['lane_counts'],
            timestamp
        )
        
        # Display results
        print(f"📈 Traffic State Summary:")
        print(f"   • Total vehicles: {traffic_state.get_total_vehicles()}")
        print(f"   • Total queue length: {traffic_state.get_total_queue_length():.1f}m")
        print(f"   • Average wait time: {traffic_state.get_average_wait_time():.1f}s")
        print(f"   • Signal phase: {traffic_state.signal_phase}")
        print(f"   • Prediction confidence: {traffic_state.prediction_confidence:.2f}")
        
        # Show vehicle type distribution
        if counting_result['total_vehicles'] > 0:
            type_dist = counter.get_vehicle_type_distribution()
            if type_dist:
                print(f"   • Vehicle types: {', '.join([f'{t}: {p:.1f}%' for t, p in type_dist.items()])}")
        
        # Simulate some special events occasionally
        if cycle == 2:
            print("\n🚨 Simulating accident event...")
            accident_detections = simulator.simulate_traffic_event('accident', timestamp)
            print(f"   Generated {len(accident_detections)} vehicles in accident backup")
        
        elif cycle == 4:
            print("\n⚡ Simulating rush hour spike...")
            rush_detections = simulator.simulate_traffic_event('rush_hour_spike', timestamp)
            print(f"   Generated {len(rush_detections)} vehicles in rush hour")
        
        # Wait before next cycle
        time.sleep(1)
    
    print("\n📊 Final System Statistics:")
    print("-" * 40)
    
    # Get traffic trends
    trends = aggregator.get_traffic_trends(hours=1)
    print(f"Traffic trend: {trends['trend_direction']}")
    print(f"Average vehicles: {trends['average_vehicles']:.1f}")
    print(f"Peak vehicles: {trends['peak_vehicles']}")
    print(f"Congestion level: {trends['congestion_level']}")
    
    # Get performance metrics
    metrics = aggregator.get_performance_metrics()
    print(f"Throughput: {metrics['throughput']:.1f} vehicles/hour")
    print(f"Efficiency: {metrics['efficiency']:.2f}")
    print(f"Average delay: {metrics['average_delay']:.1f}s")
    
    # Get vehicle counter statistics
    print(f"\nVehicle Counter Statistics:")
    for direction in ['north', 'south', 'east', 'west']:
        stats = counter.get_lane_statistics(direction, hours=1)
        if stats['total_vehicles'] > 0:
            print(f"  {direction.capitalize()}: {stats['total_vehicles']} vehicles, "
                  f"{stats['average_flow_rate']:.1f}/min flow rate")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 This demo showed:")
    print("   • Synthetic traffic data generation")
    print("   • Camera-based vehicle detection simulation")
    print("   • Vehicle counting and classification")
    print("   • Traffic data aggregation and analysis")
    print("   • Real-time traffic state monitoring")
    print("   • Performance metrics calculation")


if __name__ == "__main__":
    main()