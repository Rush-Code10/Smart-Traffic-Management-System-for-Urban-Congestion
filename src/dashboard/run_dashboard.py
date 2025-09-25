"""Dashboard launcher script for the Smart Traffic Management System."""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for running the dashboard."""
    try:
        # Create necessary directories
        directories = [
            'logs',
            'models',
            'config',
            'data'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Initialize configuration if it doesn't exist
        from config.config_manager import ConfigManager
        config_manager = ConfigManager()
        
        # Ensure we have at least one intersection configured
        if not config_manager.get_all_intersection_ids():
            logger.warning("No intersections configured, creating default intersection")
            config_manager._create_default_config()
        
        logger.info("Environment setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'torch',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are available")
    return True

def main():
    """Main function to launch the dashboard."""
    logger.info("Starting Smart Traffic Management System Dashboard")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        sys.exit(1)
    
    try:
        # Import and run the dashboard
        from dashboard.main_dashboard import main as run_dashboard
        
        logger.info("Launching Streamlit dashboard...")
        logger.info("Dashboard will be available at: http://localhost:8501")
        
        # Run the dashboard
        run_dashboard()
        
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()