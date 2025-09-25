"""Integration test for the Smart Traffic Management System Dashboard."""

import sys
import os
from pathlib import Path
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

def test_dashboard_components():
    """Test that all dashboard components can be imported and initialized."""
    logger.info("Testing dashboard component imports...")
    
    try:
        # Test imports
        from dashboard.dashboard_components import (
            TrafficMonitor, PerformanceMetrics, ManualOverride, 
            AnalyticsSection, SystemIntegrator
        )
        from config.config_manager import ConfigManager
        from models.traffic_state import TrafficState
        from agents.signal_control_manager import IntersectionSignalState, PhaseState, SignalState
        
        logger.info("✓ All imports successful")
        
        # Test component initialization
        traffic_monitor = TrafficMonitor()
        performance_metrics = PerformanceMetrics()
        manual_override = ManualOverride()
        analytics_section = AnalyticsSection()
        
        logger.info("✓ All components initialized successfully")
        
        # Test config manager
        config_manager = ConfigManager()
        logger.info("✓ Config manager initialized")
        
        # Test system integrator
        system_integrator = SystemIntegrator(config_manager)
        logger.info("✓ System integrator initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Component test failed: {e}")
        return False

def test_data_models():
    """Test that data models work correctly."""
    logger.info("Testing data models...")
    
    try:
        from models.traffic_state import TrafficState
        from models.signal_action import SignalAction
        
        # Test TrafficState creation
        traffic_state = TrafficState(
            intersection_id="test_001",
            timestamp=datetime.now(),
            vehicle_counts={'north': 15, 'south': 12, 'east': 8, 'west': 10},
            queue_lengths={'north': 25.0, 'south': 20.0, 'east': 15.0, 'west': 18.0},
            wait_times={'north': 45.0, 'south': 38.0, 'east': 30.0, 'west': 35.0},
            signal_phase='north_south_green',
            prediction_confidence=0.85
        )
        
        # Test methods
        total_vehicles = traffic_state.get_total_vehicles()
        total_queue = traffic_state.get_total_queue_length()
        avg_wait = traffic_state.get_average_wait_time()
        
        logger.info(f"✓ TrafficState: {total_vehicles} vehicles, {total_queue:.1f}m queue, {avg_wait:.1f}s wait")
        
        # Test SignalAction creation
        signal_action = SignalAction(
            intersection_id="test_001",
            phase_adjustments={'north_south_green': 10, 'east_west_green': -5},
            reasoning="Adjusting for traffic imbalance"
        )
        
        logger.info("✓ SignalAction created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data model test failed: {e}")
        return False

def test_system_integration():
    """Test system integration functionality."""
    logger.info("Testing system integration...")
    
    try:
        from dashboard.dashboard_components import SystemIntegrator
        from config.config_manager import ConfigManager
        
        # Initialize components
        config_manager = ConfigManager()
        system_integrator = SystemIntegrator(config_manager)
        
        # Test getting intersection IDs
        intersection_ids = config_manager.get_all_intersection_ids()
        logger.info(f"✓ Found {len(intersection_ids)} configured intersections")
        
        if intersection_ids:
            test_intersection = intersection_ids[0]
            
            # Test getting system status
            status = system_integrator.get_system_status(test_intersection)
            logger.info(f"✓ System status: {status['overall_status']}")
            
            # Test getting traffic state
            traffic_state = system_integrator.get_current_traffic_state(test_intersection)
            logger.info(f"✓ Traffic state: {traffic_state.get_total_vehicles()} vehicles")
            
            # Test getting signal state
            signal_state = system_integrator.get_current_signal_state(test_intersection)
            logger.info(f"✓ Signal state: {signal_state.current_cycle_phase}")
            
            # Test performance metrics
            metrics = system_integrator.get_performance_metrics(test_intersection)
            logger.info(f"✓ Performance metrics: {metrics['throughput']:.1f} veh/hr")
            
            # Test traffic trends
            trends = system_integrator.get_traffic_trends(test_intersection)
            logger.info(f"✓ Traffic trends: {trends['trend_direction']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ System integration test failed: {e}")
        return False

def test_dashboard_main():
    """Test that the main dashboard can be imported."""
    logger.info("Testing main dashboard import...")
    
    try:
        from dashboard.main_dashboard import SmartTrafficDashboard
        logger.info("✓ Main dashboard imported successfully")
        
        # Note: We can't fully test the dashboard without Streamlit context
        # but we can verify the class can be imported
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Main dashboard test failed: {e}")
        return False

def test_prediction_engine():
    """Test prediction engine functionality."""
    logger.info("Testing prediction engine...")
    
    try:
        from processors.prediction_engine import PredictionEngine, PredictionResult
        from models.traffic_state import TrafficState
        
        # Initialize prediction engine
        prediction_engine = PredictionEngine()
        logger.info("✓ Prediction engine initialized")
        
        # Create sample traffic states
        traffic_states = []
        for i in range(20):
            state = TrafficState(
                intersection_id="test_001",
                timestamp=datetime.now() - timedelta(minutes=i*5),
                vehicle_counts={'north': 10+i, 'south': 8+i, 'east': 6+i, 'west': 7+i},
                queue_lengths={'north': 15.0+i, 'south': 12.0+i, 'east': 10.0+i, 'west': 11.0+i},
                wait_times={'north': 30.0+i, 'south': 25.0+i, 'east': 20.0+i, 'west': 22.0+i},
                signal_phase='north_south_green',
                prediction_confidence=0.8
            )
            traffic_states.append(state)
        
        # Test fallback prediction (since model isn't trained)
        prediction = prediction_engine.get_fallback_prediction("test_001", traffic_states[-12:])
        logger.info(f"✓ Fallback prediction: {len(prediction.predictions)} predictions, confidence: {prediction.confidence}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Prediction engine test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting Smart Traffic Management System Dashboard Integration Tests")
    logger.info("=" * 70)
    
    tests = [
        ("Component Imports", test_dashboard_components),
        ("Data Models", test_data_models),
        ("System Integration", test_system_integration),
        ("Main Dashboard", test_dashboard_main),
        ("Prediction Engine", test_prediction_engine)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total:  {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Dashboard is ready to use.")
        logger.info("\nTo start the dashboard:")
        logger.info("  streamlit run src/dashboard/main_dashboard.py")
        logger.info("  or")
        logger.info("  python src/dashboard/run_dashboard.py")
        return True
    else:
        logger.error(f"❌ {failed} tests failed. Please check the errors above.")
        return False

def main():
    """Main function for integration tests."""
    print("Smart Traffic Management System - Dashboard Integration Tests")
    print("=" * 65)
    print()
    
    success = run_integration_tests()
    
    if success:
        print("\n🎉 Integration tests completed successfully!")
        print("The dashboard is ready to use.")
    else:
        print("\n❌ Some integration tests failed.")
        print("Please review the logs and fix any issues before using the dashboard.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())