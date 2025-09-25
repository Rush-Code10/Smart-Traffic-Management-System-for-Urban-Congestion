"""Simple launcher script for the Smart Traffic Management System Dashboard."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    print("🚦 Smart Traffic Management System Dashboard")
    print("=" * 50)
    
    # Get the path to the dashboard
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "simple_dashboard.py"
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        print(f"Expected location: {dashboard_path}")
        return 1
    
    print("🚀 Starting dashboard...")
    print(f"📁 Dashboard location: {dashboard_path}")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except FileNotFoundError:
        print("❌ Streamlit not found! Please install it with: pip install streamlit")
        return 1

if __name__ == "__main__":
    exit(main())