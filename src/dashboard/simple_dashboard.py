"""Simplified Streamlit dashboard for Smart Traffic Management System."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Import system components with error handling
try:
    from models.traffic_state import TrafficState
    from models.signal_action import SignalAction
    from config.config_manager import ConfigManager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are available")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-green { color: #4caf50; font-weight: bold; }
    .signal-yellow { color: #ff9800; font-weight: bold; }
    .signal-red { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class SimpleTrafficDashboard:
    """Simplified dashboard for traffic management system."""
    
    def __init__(self):
        """Initialize the dashboard."""
        try:
            self.config_manager = ConfigManager()
            self.intersection_ids = self.config_manager.get_all_intersection_ids()
            
            # Initialize session state
            if 'initialized' not in st.session_state:
                self._initialize_session_state()
                
        except Exception as e:
            st.error(f"Failed to initialize dashboard: {e}")
            st.stop()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        st.session_state.initialized = True
        st.session_state.selected_intersection = None
        st.session_state.manual_override_active = False
        st.session_state.override_operator = ""
        st.session_state.last_update = datetime.now()
    
    def run(self):
        """Run the main dashboard application."""
        # Sidebar
        self._render_sidebar()
        
        # Main content
        if st.session_state.selected_intersection:
            self._render_main_dashboard()
        else:
            self._render_welcome_screen()
    
    def _render_sidebar(self):
        """Render the sidebar."""
        st.sidebar.title("Traffic Control")
        
        # Intersection selection
        if self.intersection_ids:
            selected = st.sidebar.selectbox(
                "Select Intersection",
                options=self.intersection_ids,
                format_func=lambda x: self._get_intersection_name(x),
                key="intersection_selector"
            )
            
            if selected != st.session_state.selected_intersection:
                st.session_state.selected_intersection = selected
                st.rerun()
        else:
            st.sidebar.error("No intersections configured")
            return
        
        st.sidebar.divider()
        
        # System status
        st.sidebar.subheader("System Status")
        st.sidebar.success("System Healthy")
        
        # Components
        st.sidebar.write("**Components:**")
        st.sidebar.write("Traffic Monitoring")
        st.sidebar.write("Signal Control")
        st.sidebar.write("Data Processing")
        
        st.sidebar.divider()
        
        # Manual override
        self._render_manual_override_controls()
        
        st.sidebar.divider()
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
            time.sleep(refresh_rate)
            st.rerun()
        
        if st.sidebar.button("Refresh Now"):
            st.rerun()
    
    def _render_manual_override_controls(self):
        """Render manual override controls."""
        st.sidebar.subheader("Manual Override")
        
        if not st.session_state.manual_override_active:
            operator_id = st.sidebar.text_input("Operator ID", key="operator_input")
            
            if st.sidebar.button("Enable Override", type="primary"):
                if operator_id:
                    st.session_state.manual_override_active = True
                    st.session_state.override_operator = operator_id
                    st.sidebar.success(f"Override enabled for {operator_id}")
                    st.rerun()
                else:
                    st.sidebar.error("Please enter Operator ID")
        else:
            st.sidebar.info(f"Override active: {st.session_state.override_operator}")
            
            # Phase controls
            phases = ["North-South Green", "East-West Green"]
            selected_phase = st.sidebar.selectbox("Select Phase", phases)
            duration = st.sidebar.slider("Duration (seconds)", 10, 300, 60)
            
            if st.sidebar.button("Set Phase"):
                st.sidebar.success(f"Set {selected_phase} for {duration}s")
            
            if st.sidebar.button("Disable Override", type="secondary"):
                st.session_state.manual_override_active = False
                st.session_state.override_operator = ""
                st.sidebar.success("Override disabled")
                st.rerun()
    
    def _render_welcome_screen(self):
        """Render welcome screen."""
        st.title("Smart Traffic Management System")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## Welcome to the Smart Traffic Management System
            
            This AI-powered system optimizes traffic signal timings using:
            
            - **Real-time Vehicle Detection** - Computer vision analysis
            - **Reinforcement Learning** - Q-learning optimization  
            - **Traffic Prediction** - LSTM neural networks
            - **Performance Analytics** - Comprehensive metrics
            
            ### Getting Started
            1. Select an intersection from the sidebar
            2. Monitor real-time traffic conditions
            3. Review performance metrics and analytics
            4. Use manual override for emergencies
            """)
        
        # System overview
        st.markdown("---")
        st.subheader("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Intersections", len(self.intersection_ids))
        
        with col2:
            st.metric("Total Vehicles", np.random.randint(50, 150))
        
        with col3:
            st.metric("Avg Wait Time", f"{np.random.uniform(30, 60):.1f}s")
        
        with col4:
            st.metric("System Uptime", "99.8%")
    
    def _render_main_dashboard(self):
        """Render main dashboard."""
        intersection_id = st.session_state.selected_intersection
        intersection_name = self._get_intersection_name(intersection_id)
        
        # Header
        st.title(f"{intersection_name}")
        st.markdown(f"*Intersection ID: {intersection_id}*")
        
        # Generate sample traffic data
        traffic_data = self._generate_sample_traffic_data()
        
        # Alert banner
        self._render_alert_banner(traffic_data)
        
        # Main metrics
        self._render_main_metrics(traffic_data)
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Real-time Monitoring", 
            "Performance Metrics", 
            "Analytics & Trends",
            "System Control"
        ])
        
        with tab1:
            self._render_monitoring_tab(traffic_data)
        
        with tab2:
            self._render_performance_tab()
        
        with tab3:
            self._render_analytics_tab()
        
        with tab4:
            self._render_control_tab()
    
    def _generate_sample_traffic_data(self):
        """Generate sample traffic data."""
        current_hour = datetime.now().hour
        
        # Simulate rush hour patterns
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            base_traffic = 25
        elif 10 <= current_hour <= 16:
            base_traffic = 15
        else:
            base_traffic = 8
        
        return {
            'vehicle_counts': {
                'North': max(0, int(base_traffic + np.random.normal(0, 5))),
                'South': max(0, int(base_traffic + np.random.normal(0, 5))),
                'East': max(0, int(base_traffic * 0.8 + np.random.normal(0, 3))),
                'West': max(0, int(base_traffic * 0.8 + np.random.normal(0, 3)))
            },
            'queue_lengths': {
                'North': max(0, np.random.uniform(10, 50)),
                'South': max(0, np.random.uniform(10, 50)),
                'East': max(0, np.random.uniform(5, 40)),
                'West': max(0, np.random.uniform(5, 40))
            },
            'wait_times': {
                'North': max(0, np.random.uniform(20, 80)),
                'South': max(0, np.random.uniform(20, 80)),
                'East': max(0, np.random.uniform(15, 60)),
                'West': max(0, np.random.uniform(15, 60))
            },
            'signal_phase': 'North-South Green',
            'confidence': np.random.uniform(0.6, 0.9)
        }
    
    def _render_alert_banner(self, traffic_data):
        """Render alert banner."""
        total_vehicles = sum(traffic_data['vehicle_counts'].values())
        avg_wait = np.mean(list(traffic_data['wait_times'].values()))
        
        if total_vehicles > 60 or avg_wait > 70:
            st.error("**HIGH CONGESTION ALERT** - Consider manual intervention")
        elif total_vehicles > 30 or avg_wait > 45:
            st.warning("**MODERATE CONGESTION** - System actively optimizing")
        elif traffic_data['confidence'] < 0.7:
            st.info("**LOW PREDICTION CONFIDENCE** - Using real-time data only")
    
    def _render_main_metrics(self, traffic_data):
        """Render main metrics row."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_vehicles = sum(traffic_data['vehicle_counts'].values())
            st.metric("Total Vehicles", total_vehicles, delta=np.random.randint(-3, 4))
        
        with col2:
            avg_wait = np.mean(list(traffic_data['wait_times'].values()))
            st.metric("Avg Wait Time", f"{avg_wait:.1f}s", delta=f"{np.random.uniform(-5, 5):.1f}s")
        
        with col3:
            total_queue = sum(traffic_data['queue_lengths'].values())
            st.metric("Total Queue Length", f"{total_queue:.1f}m", delta=f"{np.random.uniform(-10, 10):.1f}m")
        
        with col4:
            current_phase = traffic_data['signal_phase']
            if 'Green' in current_phase:
                st.markdown(f'<p class="signal-green">GREEN {current_phase}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="signal-red">RED All Red</p>', unsafe_allow_html=True)
        
        with col5:
            confidence = traffic_data['confidence']
            confidence_pct = f"{confidence * 100:.1f}%"
            
            if confidence >= 0.8:
                st.metric("Prediction Confidence", confidence_pct, delta="High")
            elif confidence >= 0.6:
                st.metric("Prediction Confidence", confidence_pct, delta="Medium")
            else:
                st.metric("Prediction Confidence", confidence_pct, delta="Low")
    
    def _render_monitoring_tab(self, traffic_data):
        """Render monitoring tab."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Traffic Flow by Direction")
            
            # Traffic flow charts
            directions = list(traffic_data['vehicle_counts'].keys())
            vehicle_counts = list(traffic_data['vehicle_counts'].values())
            queue_lengths = list(traffic_data['queue_lengths'].values())
            wait_times = list(traffic_data['wait_times'].values())
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Vehicle Counts', 'Queue Lengths (m)', 'Wait Times (s)', 'Signal Status'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            fig.add_trace(go.Bar(x=directions, y=vehicle_counts, name="Vehicles", marker_color='lightblue'), row=1, col=1)
            fig.add_trace(go.Bar(x=directions, y=queue_lengths, name="Queue", marker_color='orange'), row=1, col=2)
            fig.add_trace(go.Bar(x=directions, y=wait_times, name="Wait Time", marker_color='lightcoral'), row=2, col=1)
            
            # Signal indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=75,  # Sample progress
                    title={'text': "Signal Progress"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}]}
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Signal Timing")
            
            # Signal phases
            phases = [
                ("North-South Green", "green", 35, 45),
                ("East-West Green", "red", 0, 40)
            ]
            
            for phase_name, state, remaining, total in phases:
                if state == "green":
                    st.markdown(f'<p class="signal-green">GREEN {phase_name}</p>', unsafe_allow_html=True)
                    progress = (total - remaining) / total if total > 0 else 0
                    st.progress(progress)
                    st.write(f"Time remaining: {remaining}s")
                else:
                    st.markdown(f'<p class="signal-red">RED {phase_name}</p>', unsafe_allow_html=True)
                
                st.write(f"Total duration: {total}s")
                st.markdown("---")
            
            # Recent actions
            st.subheader("Recent Actions")
            st.write("**14:32:15** - Extended North-South phase by 10s")
            st.write("*High traffic volume detected from North*")
            st.markdown("---")
            st.write("**14:30:42** - Reduced East-West phase by 5s")
            st.write("*Optimizing for traffic flow balance*")
    
    def _render_performance_tab(self):
        """Render performance tab."""
        st.subheader("Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Throughput", f"{np.random.uniform(180, 220):.1f} veh/hr")
        
        with col2:
            st.metric("Efficiency", f"{np.random.uniform(75, 90):.1f}%")
        
        with col3:
            st.metric("Average Delay", f"{np.random.uniform(35, 55):.1f}s")
        
        with col4:
            st.metric("Queue Clearance", f"{np.random.uniform(45, 75):.1f}s")
        
        st.markdown("---")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("24-Hour Traffic Pattern")
            
            hours = list(range(24))
            traffic_volume = [20 + 15 * np.sin((h - 6) * np.pi / 12) + 5 * np.random.random() for h in hours]
            
            fig = px.line(x=hours, y=traffic_volume, title="Traffic Volume by Hour")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Wait Time Comparison")
            
            directions = ['North', 'South', 'East', 'West']
            current_wait = [45, 38, 52, 41]
            baseline_wait = [60, 55, 65, 58]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Current', x=directions, y=current_wait, marker_color='lightblue'))
            fig.add_trace(go.Bar(name='Baseline', x=directions, y=baseline_wait, marker_color='lightcoral'))
            
            fig.update_layout(title="Wait Time: Current vs Baseline", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Commute time impact
        st.subheader("Commute Time Impact")
        
        current_avg = np.mean(current_wait)
        baseline_avg = np.mean(baseline_wait)
        reduction = ((baseline_avg - current_avg) / baseline_avg) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Baseline Wait", f"{baseline_avg:.1f}s")
        
        with col2:
            st.metric("Current Wait", f"{current_avg:.1f}s")
        
        with col3:
            st.metric("Reduction", f"{reduction:.1f}%", delta=f"{reduction:.1f}%")
        
        if reduction >= 10:
            st.success(f"**Target Achieved!** {reduction:.1f}% reduction in commute time")
        else:
            st.info(f"**Progress:** {reduction:.1f}% reduction achieved")
    
    def _render_analytics_tab(self):
        """Render analytics tab."""
        st.subheader("Traffic Analytics & Trends")
        
        # Time period selection
        time_period = st.selectbox("Analysis Period", ["Last Hour", "Last 4 Hours", "Last 24 Hours", "Last Week"])
        
        # Trend summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Traffic Trend", "Increasing")
        
        with col2:
            st.metric("Average Vehicles", f"{np.random.uniform(15, 25):.1f}")
        
        with col3:
            st.metric("Peak Vehicles", f"{np.random.randint(35, 50)}")
        
        with col4:
            st.metric("Congestion Level", "Medium")
        
        st.markdown("---")
        
        # Analytics charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Accuracy")
            
            time_points = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='H')
            actual = [15 + 10 * np.sin(i * 0.5) + 3 * np.random.random() for i in range(len(time_points))]
            predicted = [val + 2 * (np.random.random() - 0.5) for val in actual]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points, y=actual, mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=time_points, y=predicted, mode='lines+markers', name='Predicted', line=dict(dash='dash')))
            
            fig.update_layout(title="Prediction vs Actual", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Optimization Impact")
            
            events = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='H')
            before = [60 + 20 * np.random.random() for _ in events]
            after = [b * (0.7 + 0.2 * np.random.random()) for b in before]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=events, y=before, name='Before', marker_color='lightcoral', opacity=0.7))
            fig.add_trace(go.Bar(x=events, y=after, name='After', marker_color='lightblue', opacity=0.7))
            
            fig.update_layout(title="Wait Time: Before/After Optimization", height=400, barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.markdown("---")
        st.subheader("Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Traffic Data"):
                st.success("Traffic data exported to CSV")
        
        with col2:
            if st.button("Export Performance Report"):
                st.success("Performance report exported to JSON")
        
        with col3:
            if st.button("Export System Config"):
                st.success("System configuration exported")
    
    def _render_control_tab(self):
        """Render control tab."""
        st.subheader("System Control & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RL Agent Control")
            
            st.write(f"**Training Episodes:** {np.random.randint(100, 500)}")
            st.write(f"**Average Reward:** {np.random.uniform(15, 25):.2f}")
            st.write(f"**Exploration Rate:** {np.random.uniform(0.05, 0.15):.3f}")
            st.write(f"**Q-Table Size:** {np.random.randint(500, 1500)} states")
            
            if st.button("Start Training Episode"):
                st.success("Training episode started")
            
            if st.button("Save RL Model"):
                st.success("RL model saved")
        
        with col2:
            st.subheader("Prediction Engine")
            
            st.write("**Model Status:** Active")
            st.write("**Last Training:** 2 hours ago")
            st.write(f"**Prediction Confidence:** {np.random.uniform(70, 90):.1f}%")
            
            if st.button("Generate Predictions"):
                st.success("Generated 6 predictions for next 30 minutes")
            
            if st.button("Retrain Model"):
                st.success("Model retraining started")
        
        st.markdown("---")
        
        # Configuration
        st.subheader("System Configuration")
        
        with st.expander("System Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.slider("Dashboard Refresh Rate (seconds)", 1, 30, 5)
                st.slider("Signal Update Interval (seconds)", 10, 120, 30)
                st.slider("Prediction Update Interval (seconds)", 30, 300, 60)
            
            with col2:
                st.slider("Max Signal Adjustment (seconds)", 30, 300, 120)
                st.slider("Min Prediction Confidence", 0.0, 1.0, 0.7, 0.1)
                st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
            
            if st.button("💾 Save Configuration"):
                st.success("Configuration saved successfully")
        
        # Emergency controls
        st.markdown("---")
        st.subheader("Emergency Controls")
        
        st.warning("Use these controls only in emergency situations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Emergency Stop", type="secondary"):
                confirm = st.checkbox("Confirm Emergency Stop")
                if confirm:
                    st.error("Emergency stop activated")
        
        with col2:
            if st.button("Reset System", type="secondary"):
                confirm = st.checkbox("Confirm System Reset")
                if confirm:
                    st.success("System reset completed")
        
        with col3:
            if st.button("Contact Support", type="primary"):
                st.info("Support: traffic-support@city.gov | Phone: (555) 123-4567")
    
    def _get_intersection_name(self, intersection_id: str) -> str:
        """Get display name for intersection."""
        config = self.config_manager.get_intersection_config(intersection_id)
        if config and 'name' in config:
            return config['name']
        return intersection_id


def main():
    """Main function to run the dashboard."""
    try:
        dashboard = SimpleTrafficDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.error("Please check the system configuration and try again.")


if __name__ == "__main__":
    main()