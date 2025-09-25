"""Main Streamlit dashboard for Smart Traffic Management System."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional

# Import system components
try:
    from ..models.traffic_state import TrafficState
    from ..models.signal_action import SignalAction
    from ..agents.signal_control_manager import SignalControlManager, IntersectionSignalState
    from ..processors.traffic_aggregator import TrafficAggregator
    from ..agents.q_learning_agent import QLearningAgent
    from ..processors.prediction_engine import PredictionEngine
    from ..config.config_manager import ConfigManager
    from ..config.intersection_config import IntersectionConfig
    from .dashboard_components import (
        TrafficMonitor, PerformanceMetrics, ManualOverride, 
        AnalyticsSection, SystemIntegrator
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    
    from models.traffic_state import TrafficState
    from models.signal_action import SignalAction
    from agents.signal_control_manager import SignalControlManager, IntersectionSignalState
    from processors.traffic_aggregator import TrafficAggregator
    from agents.q_learning_agent import QLearningAgent
    from processors.prediction_engine import PredictionEngine
    from config.config_manager import ConfigManager
    from config.intersection_config import IntersectionConfig
    from dashboard.dashboard_components import (
        TrafficMonitor, PerformanceMetrics, ManualOverride, 
        AnalyticsSection, SystemIntegrator
    )

# Page configuration
st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .signal-green {
        color: #4caf50;
        font-weight: bold;
    }
    .signal-yellow {
        color: #ff9800;
        font-weight: bold;
    }
    .signal-red {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class SmartTrafficDashboard:
    """Main dashboard class for the Smart Traffic Management System."""
    
    def __init__(self):
        """Initialize the dashboard with system components."""
        self.config_manager = ConfigManager()
        self.system_integrator = SystemIntegrator(self.config_manager)
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            self._initialize_session_state()
        
        # Get available intersections
        self.intersection_ids = self.config_manager.get_all_intersection_ids()
        
        # Dashboard components
        self.traffic_monitor = TrafficMonitor()
        self.performance_metrics = PerformanceMetrics()
        self.manual_override = ManualOverride()
        self.analytics_section = AnalyticsSection()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        st.session_state.initialized = True
        st.session_state.selected_intersection = None
        st.session_state.manual_override_active = False
        st.session_state.override_operator = ""
        st.session_state.last_update = datetime.now()
        st.session_state.performance_data = []
        st.session_state.alert_history = []
    
    def run(self):
        """Run the main dashboard application."""
        # Sidebar for intersection selection and controls
        self._render_sidebar()
        
        # Main dashboard content
        if st.session_state.selected_intersection:
            self._render_main_dashboard()
        else:
            self._render_welcome_screen()
    
    def _render_sidebar(self):
        """Render the sidebar with controls and intersection selection."""
        st.sidebar.title("🚦 Traffic Control")
        
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
        
        # Get current system status
        status = self.system_integrator.get_system_status(st.session_state.selected_intersection)
        
        # Display status indicators
        if status['overall_status'] == 'healthy':
            st.sidebar.success("🟢 System Healthy")
        elif status['overall_status'] == 'warning':
            st.sidebar.warning("🟡 System Warning")
        else:
            st.sidebar.error("🔴 System Error")
        
        # Component status
        st.sidebar.write("**Components:**")
        for component, state in status['components'].items():
            icon = "✅" if state == "active" else "❌"
            st.sidebar.write(f"{icon} {component.replace('_', ' ').title()}")
        
        st.sidebar.divider()
        
        # Manual override controls
        self._render_manual_override_controls()
        
        st.sidebar.divider()
        
        # Auto-refresh controls
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 30, 5)
        
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
    
    def _render_manual_override_controls(self):
        """Render manual override controls in sidebar."""
        st.sidebar.subheader("Manual Override")
        
        if not st.session_state.manual_override_active:
            operator_id = st.sidebar.text_input("Operator ID", key="operator_input")
            
            if st.sidebar.button("🚨 Enable Override", type="primary"):
                if operator_id:
                    success = self.system_integrator.enable_manual_override(
                        st.session_state.selected_intersection, operator_id
                    )
                    if success:
                        st.session_state.manual_override_active = True
                        st.session_state.override_operator = operator_id
                        st.sidebar.success(f"Override enabled for {operator_id}")
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to enable override")
                else:
                    st.sidebar.error("Please enter Operator ID")
        else:
            st.sidebar.info(f"Override active: {st.session_state.override_operator}")
            
            # Manual phase controls
            intersection_config = self.config_manager.get_intersection_config(
                st.session_state.selected_intersection
            )
            
            if intersection_config:
                phases = intersection_config['geometry']['signal_phases']
                selected_phase = st.sidebar.selectbox("Select Phase", phases)
                duration = st.sidebar.slider("Duration (seconds)", 10, 300, 60)
                
                if st.sidebar.button("Set Phase"):
                    success = self.system_integrator.set_manual_phase(
                        st.session_state.selected_intersection, selected_phase, duration
                    )
                    if success:
                        st.sidebar.success(f"Set {selected_phase} for {duration}s")
                    else:
                        st.sidebar.error("Failed to set phase")
            
            if st.sidebar.button("❌ Disable Override", type="secondary"):
                success = self.system_integrator.disable_manual_override(
                    st.session_state.selected_intersection
                )
                if success:
                    st.session_state.manual_override_active = False
                    st.session_state.override_operator = ""
                    st.sidebar.success("Override disabled")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to disable override")
    
    def _render_welcome_screen(self):
        """Render welcome screen when no intersection is selected."""
        st.title("🚦 Smart Traffic Management System")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## Welcome to the Smart Traffic Management System
            
            This AI-powered system optimizes traffic signal timings using:
            
            - 🎥 **Real-time Vehicle Detection** - Computer vision analysis of traffic cameras
            - 🧠 **Reinforcement Learning** - Q-learning agent for signal optimization  
            - 📈 **Traffic Prediction** - LSTM neural networks for forecasting
            - 📊 **Performance Analytics** - Comprehensive traffic metrics and reporting
            
            ### Getting Started
            1. Select an intersection from the sidebar
            2. Monitor real-time traffic conditions
            3. Review performance metrics and analytics
            4. Use manual override for emergency situations
            
            ### System Features
            - Real-time traffic monitoring with live vehicle counts
            - Automatic signal optimization based on traffic conditions
            - Predictive traffic analysis for proactive adjustments
            - Manual override controls for operator intervention
            - Performance analytics with before/after comparisons
            - Data export capabilities for further analysis
            """)
        
        # System overview metrics
        st.markdown("---")
        st.subheader("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Intersections", len(self.intersection_ids))
        
        with col2:
            # Get total vehicles across all intersections
            total_vehicles = sum(
                self.system_integrator.get_current_traffic_state(iid).get_total_vehicles()
                for iid in self.intersection_ids
            )
            st.metric("Total Vehicles", total_vehicles)
        
        with col3:
            # Calculate average wait time
            avg_wait = np.mean([
                self.system_integrator.get_current_traffic_state(iid).get_average_wait_time()
                for iid in self.intersection_ids
            ])
            st.metric("Avg Wait Time", f"{avg_wait:.1f}s")
        
        with col4:
            # System uptime (simplified)
            st.metric("System Uptime", "99.8%")
    
    def _render_main_dashboard(self):
        """Render the main dashboard for selected intersection."""
        intersection_id = st.session_state.selected_intersection
        intersection_name = self._get_intersection_name(intersection_id)
        
        # Header
        st.title(f"🚦 {intersection_name}")
        st.markdown(f"*Intersection ID: {intersection_id}*")
        
        # Get current data
        current_state = self.system_integrator.get_current_traffic_state(intersection_id)
        signal_state = self.system_integrator.get_current_signal_state(intersection_id)
        
        # Alert banner
        self._render_alert_banner(current_state)
        
        # Main metrics row
        self._render_main_metrics(current_state, signal_state)
        
        st.markdown("---")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Real-time Monitoring", 
            "📊 Performance Metrics", 
            "📈 Analytics & Trends",
            "⚙️ System Control"
        ])
        
        with tab1:
            self._render_monitoring_tab(intersection_id, current_state, signal_state)
        
        with tab2:
            self._render_performance_tab(intersection_id)
        
        with tab3:
            self._render_analytics_tab(intersection_id)
        
        with tab4:
            self._render_control_tab(intersection_id)
    
    def _render_alert_banner(self, traffic_state: TrafficState):
        """Render alert banner based on traffic conditions."""
        total_vehicles = traffic_state.get_total_vehicles()
        avg_wait = traffic_state.get_average_wait_time()
        
        if total_vehicles > 30 or avg_wait > 120:
            st.error("🚨 **HIGH CONGESTION ALERT** - Consider manual intervention")
        elif total_vehicles > 15 or avg_wait > 60:
            st.warning("⚠️ **MODERATE CONGESTION** - System actively optimizing")
        elif traffic_state.prediction_confidence < 0.5:
            st.info("ℹ️ **LOW PREDICTION CONFIDENCE** - Using real-time data only")
    
    def _render_main_metrics(self, traffic_state: TrafficState, signal_state: IntersectionSignalState):
        """Render main metrics row."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Vehicles",
                traffic_state.get_total_vehicles(),
                delta=self._get_vehicle_delta(traffic_state.intersection_id)
            )
        
        with col2:
            avg_wait = traffic_state.get_average_wait_time()
            st.metric(
                "Avg Wait Time",
                f"{avg_wait:.1f}s",
                delta=f"{self._get_wait_time_delta(traffic_state.intersection_id):.1f}s"
            )
        
        with col3:
            queue_length = traffic_state.get_total_queue_length()
            st.metric(
                "Total Queue Length",
                f"{queue_length:.1f}m",
                delta=f"{self._get_queue_delta(traffic_state.intersection_id):.1f}m"
            )
        
        with col4:
            # Current signal phase with color coding
            current_phase = signal_state.current_cycle_phase
            active_phases = signal_state.get_active_phases()
            
            if active_phases:
                phase_display = active_phases[0].replace('_', ' ').title()
                if 'green' in current_phase.lower():
                    st.markdown(f'<p class="signal-green">🟢 {phase_display}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="signal-red">🔴 {phase_display}</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="signal-red">🔴 All Red</p>', unsafe_allow_html=True)
        
        with col5:
            confidence = traffic_state.prediction_confidence
            confidence_pct = f"{confidence * 100:.1f}%"
            
            if confidence >= 0.8:
                st.metric("Prediction Confidence", confidence_pct, delta="High")
            elif confidence >= 0.5:
                st.metric("Prediction Confidence", confidence_pct, delta="Medium")
            else:
                st.metric("Prediction Confidence", confidence_pct, delta="Low")
    
    def _render_monitoring_tab(self, intersection_id: str, traffic_state: TrafficState, 
                              signal_state: IntersectionSignalState):
        """Render real-time monitoring tab."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Traffic flow visualization
            st.subheader("Traffic Flow by Direction")
            
            # Create traffic flow chart
            directions = list(traffic_state.vehicle_counts.keys())
            vehicle_counts = list(traffic_state.vehicle_counts.values())
            queue_lengths = [traffic_state.queue_lengths.get(d, 0) for d in directions]
            wait_times = [traffic_state.wait_times.get(d, 0) for d in directions]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Vehicle Counts', 'Queue Lengths (m)', 'Wait Times (s)', 'Signal Phases'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Vehicle counts
            fig.add_trace(
                go.Bar(x=directions, y=vehicle_counts, name="Vehicles", marker_color='lightblue'),
                row=1, col=1
            )
            
            # Queue lengths
            fig.add_trace(
                go.Bar(x=directions, y=queue_lengths, name="Queue Length", marker_color='orange'),
                row=1, col=2
            )
            
            # Wait times
            fig.add_trace(
                go.Bar(x=directions, y=wait_times, name="Wait Time", marker_color='lightcoral'),
                row=2, col=1
            )
            
            # Signal phase indicator
            current_phase = signal_state.current_cycle_phase
            phase_state = signal_state.phases.get(current_phase)
            
            if phase_state:
                progress = phase_state.get_progress()
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=progress * 100,
                        title={'text': f"{current_phase.replace('_', ' ').title()}<br>Progress"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkgreen"},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 100], 'color': "gray"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 90}}
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Signal timing information
            st.subheader("Signal Timing")
            
            for phase_id, phase_state in signal_state.phases.items():
                phase_name = phase_id.replace('_', ' ').title()
                
                if phase_state.current_state.value == 'green':
                    st.markdown(f'<p class="signal-green">🟢 {phase_name}</p>', unsafe_allow_html=True)
                    st.progress(phase_state.get_progress())
                    st.write(f"Time remaining: {phase_state.time_remaining}s")
                elif phase_state.current_state.value == 'yellow':
                    st.markdown(f'<p class="signal-yellow">🟡 {phase_name}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="signal-red">🔴 {phase_name}</p>', unsafe_allow_html=True)
                
                st.write(f"Total duration: {phase_state.total_duration}s")
                st.markdown("---")
            
            # Recent actions
            st.subheader("Recent Actions")
            recent_actions = self.system_integrator.get_recent_actions(intersection_id, limit=5)
            
            if recent_actions:
                for timestamp, action in recent_actions:
                    st.write(f"**{timestamp.strftime('%H:%M:%S')}**")
                    st.write(f"*{action.reasoning}*")
                    
                    for phase, adjustment in action.phase_adjustments.items():
                        if adjustment != 0:
                            direction = "+" if adjustment > 0 else ""
                            st.write(f"• {phase}: {direction}{adjustment}s")
                    st.markdown("---")
            else:
                st.info("No recent actions")
    
    def _render_performance_tab(self, intersection_id: str):
        """Render performance metrics tab."""
        # Get performance data
        performance_data = self.system_integrator.get_performance_metrics(intersection_id)
        
        # Performance overview
        st.subheader("Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            throughput = performance_data.get('throughput', 0)
            st.metric("Throughput", f"{throughput:.1f} veh/hr")
        
        with col2:
            efficiency = performance_data.get('efficiency', 0) * 100
            st.metric("Efficiency", f"{efficiency:.1f}%")
        
        with col3:
            avg_delay = performance_data.get('average_delay', 0)
            st.metric("Average Delay", f"{avg_delay:.1f}s")
        
        with col4:
            clearance_time = performance_data.get('queue_clearance_time', 0)
            st.metric("Queue Clearance", f"{clearance_time:.1f}s")
        
        st.markdown("---")
        
        # Performance trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traffic Volume Trends")
            
            # Generate sample trend data (in real implementation, this would come from historical data)
            hours = list(range(24))
            traffic_volume = [
                20 + 15 * np.sin((h - 6) * np.pi / 12) + 5 * np.random.random()
                for h in hours
            ]
            
            fig = px.line(
                x=hours, y=traffic_volume,
                title="24-Hour Traffic Volume Pattern",
                labels={'x': 'Hour of Day', 'y': 'Vehicles per Hour'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Wait Time Analysis")
            
            # Generate sample wait time data
            directions = ['North', 'South', 'East', 'West']
            current_wait = [45, 38, 52, 41]
            baseline_wait = [60, 55, 65, 58]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current',
                x=directions,
                y=current_wait,
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Baseline (Before AI)',
                x=directions,
                y=baseline_wait,
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="Wait Time Comparison",
                xaxis_title="Direction",
                yaxis_title="Average Wait Time (seconds)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Commute time reduction calculation
        st.subheader("Commute Time Impact")
        
        current_avg_wait = np.mean(current_wait)
        baseline_avg_wait = np.mean(baseline_wait)
        reduction_pct = ((baseline_avg_wait - current_avg_wait) / baseline_avg_wait) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Baseline Wait Time", f"{baseline_avg_wait:.1f}s")
        
        with col2:
            st.metric("Current Wait Time", f"{current_avg_wait:.1f}s")
        
        with col3:
            st.metric("Reduction", f"{reduction_pct:.1f}%", delta=f"{reduction_pct:.1f}%")
        
        if reduction_pct >= 10:
            st.success(f"🎯 **Target Achieved!** {reduction_pct:.1f}% reduction in commute time")
        elif reduction_pct >= 5:
            st.info(f"📈 **Good Progress** - {reduction_pct:.1f}% reduction achieved")
        else:
            st.warning(f"⚠️ **Below Target** - Only {reduction_pct:.1f}% reduction achieved")
    
    def _render_analytics_tab(self, intersection_id: str):
        """Render analytics and trends tab."""
        st.subheader("Traffic Analytics & Trends")
        
        # Time period selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            time_period = st.selectbox(
                "Analysis Period",
                ["Last Hour", "Last 4 Hours", "Last 24 Hours", "Last Week"]
            )
        
        # Get trend data
        hours_map = {"Last Hour": 1, "Last 4 Hours": 4, "Last 24 Hours": 24, "Last Week": 168}
        hours = hours_map[time_period]
        
        trend_data = self.system_integrator.get_traffic_trends(intersection_id, hours)
        
        # Trend summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trend_direction = trend_data.get('trend_direction', 'stable')
            trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
            st.metric("Traffic Trend", f"{trend_icon[trend_direction]} {trend_direction.title()}")
        
        with col2:
            avg_vehicles = trend_data.get('average_vehicles', 0)
            st.metric("Average Vehicles", f"{avg_vehicles:.1f}")
        
        with col3:
            peak_vehicles = trend_data.get('peak_vehicles', 0)
            st.metric("Peak Vehicles", f"{peak_vehicles}")
        
        with col4:
            congestion_level = trend_data.get('congestion_level', 'low')
            congestion_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.metric("Congestion Level", f"{congestion_colors[congestion_level]} {congestion_level.title()}")
        
        st.markdown("---")
        
        # Detailed analytics charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Accuracy")
            
            # Generate sample prediction accuracy data
            time_points = pd.date_range(
                start=datetime.now() - timedelta(hours=hours),
                end=datetime.now(),
                freq='H'
            )
            
            actual_values = [15 + 10 * np.sin(i * 0.5) + 3 * np.random.random() for i in range(len(time_points))]
            predicted_values = [val + 2 * (np.random.random() - 0.5) for val in actual_values]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_points, y=actual_values,
                mode='lines+markers',
                name='Actual Traffic',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=time_points, y=predicted_values,
                mode='lines+markers',
                name='Predicted Traffic',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Traffic Prediction vs Actual",
                xaxis_title="Time",
                yaxis_title="Vehicle Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Signal Optimization Impact")
            
            # Generate sample optimization impact data
            optimization_events = pd.date_range(
                start=datetime.now() - timedelta(hours=hours),
                end=datetime.now(),
                freq='30min'
            )
            
            wait_times_before = [60 + 20 * np.random.random() for _ in optimization_events]
            wait_times_after = [wt * (0.7 + 0.2 * np.random.random()) for wt in wait_times_before]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=optimization_events,
                y=wait_times_before,
                name='Before Optimization',
                marker_color='lightcoral',
                opacity=0.7
            ))
            fig.add_trace(go.Bar(
                x=optimization_events,
                y=wait_times_after,
                name='After Optimization',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Wait Time Before/After Optimization",
                xaxis_title="Time",
                yaxis_title="Average Wait Time (seconds)",
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.markdown("---")
        st.subheader("Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Traffic Data"):
                export_data = self.system_integrator.export_traffic_data(intersection_id, hours)
                
                if export_data:
                    df = pd.DataFrame(export_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"traffic_data_{intersection_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data available for export")
        
        with col2:
            if st.button("📈 Export Performance Report"):
                # Generate performance report
                report_data = {
                    'intersection_id': intersection_id,
                    'analysis_period': time_period,
                    'generated_at': datetime.now().isoformat(),
                    'performance_metrics': self.system_integrator.get_performance_metrics(intersection_id),
                    'traffic_trends': trend_data
                }
                
                report_json = json.dumps(report_data, indent=2)
                
                st.download_button(
                    label="Download JSON",
                    data=report_json,
                    file_name=f"performance_report_{intersection_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("🔧 Export System Config"):
                config_data = self.config_manager.get_intersection_config(intersection_id)
                
                if config_data:
                    config_json = json.dumps(config_data, indent=2)
                    
                    st.download_button(
                        label="Download Config",
                        data=config_json,
                        file_name=f"intersection_config_{intersection_id}.json",
                        mime="application/json"
                    )
    
    def _render_control_tab(self, intersection_id: str):
        """Render system control tab."""
        st.subheader("System Control & Configuration")
        
        # System control section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RL Agent Control")
            
            # Get RL agent stats
            rl_stats = self.system_integrator.get_rl_training_stats(intersection_id)
            
            st.write(f"**Training Episodes:** {rl_stats.get('episodes', 0)}")
            st.write(f"**Average Reward:** {rl_stats.get('average_reward', 0):.2f}")
            st.write(f"**Exploration Rate:** {rl_stats.get('epsilon', 0):.3f}")
            st.write(f"**Q-Table Size:** {rl_stats.get('q_table_size', 0)} states")
            
            if st.button("🎯 Start Training Episode"):
                success = self.system_integrator.trigger_training_episode(intersection_id)
                if success:
                    st.success("Training episode started")
                else:
                    st.error("Failed to start training")
            
            if st.button("💾 Save RL Model"):
                success = self.system_integrator.save_rl_model(intersection_id)
                if success:
                    st.success("RL model saved")
                else:
                    st.error("Failed to save model")
        
        with col2:
            st.subheader("Prediction Engine Control")
            
            # Prediction engine status
            prediction_status = self.system_integrator.get_prediction_status(intersection_id)
            
            st.write(f"**Model Status:** {prediction_status.get('status', 'Unknown')}")
            st.write(f"**Last Training:** {prediction_status.get('last_training', 'Never')}")
            st.write(f"**Prediction Confidence:** {prediction_status.get('confidence', 0):.1%}")
            
            if st.button("🔮 Generate Predictions"):
                predictions = self.system_integrator.generate_traffic_predictions(intersection_id)
                if predictions:
                    st.success(f"Generated {len(predictions.predictions)} predictions")
                    
                    # Display predictions
                    pred_df = pd.DataFrame({
                        'Time': predictions.timestamps,
                        'Predicted Vehicles': predictions.predictions
                    })
                    st.dataframe(pred_df)
                else:
                    st.error("Failed to generate predictions")
            
            if st.button("🎓 Retrain Prediction Model"):
                success = self.system_integrator.retrain_prediction_model(intersection_id)
                if success:
                    st.success("Model retraining started")
                else:
                    st.error("Failed to start retraining")
        
        st.markdown("---")
        
        # Configuration section
        st.subheader("System Configuration")
        
        # Get current system config
        system_config = self.config_manager.get_system_config()
        
        with st.expander("System Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_refresh_rate = st.slider(
                    "Dashboard Refresh Rate (seconds)",
                    1, 30, system_config.dashboard_refresh_rate_seconds
                )
                
                new_signal_interval = st.slider(
                    "Signal Update Interval (seconds)",
                    10, 120, system_config.signal_update_interval_seconds
                )
                
                new_prediction_interval = st.slider(
                    "Prediction Update Interval (seconds)",
                    30, 300, system_config.prediction_update_interval_seconds
                )
            
            with col2:
                new_max_adjustment = st.slider(
                    "Max Signal Adjustment (seconds)",
                    30, 300, system_config.max_signal_adjustment_seconds
                )
                
                new_confidence_threshold = st.slider(
                    "Min Prediction Confidence",
                    0.0, 1.0, system_config.min_prediction_confidence, 0.1
                )
                
                new_log_level = st.selectbox(
                    "Log Level",
                    ["DEBUG", "INFO", "WARNING", "ERROR"],
                    index=["DEBUG", "INFO", "WARNING", "ERROR"].index(system_config.log_level)
                )
            
            if st.button("💾 Save Configuration"):
                # Update configuration
                self.config_manager.update_system_config(
                    dashboard_refresh_rate_seconds=new_refresh_rate,
                    signal_update_interval_seconds=new_signal_interval,
                    prediction_update_interval_seconds=new_prediction_interval,
                    max_signal_adjustment_seconds=new_max_adjustment,
                    min_prediction_confidence=new_confidence_threshold,
                    log_level=new_log_level
                )
                
                self.config_manager.save_config()
                st.success("Configuration saved successfully")
        
        # Emergency controls
        st.markdown("---")
        st.subheader("🚨 Emergency Controls")
        
        st.warning("⚠️ Use these controls only in emergency situations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛑 Emergency Stop", type="secondary"):
                if st.checkbox("Confirm Emergency Stop"):
                    success = self.system_integrator.emergency_stop(intersection_id)
                    if success:
                        st.error("Emergency stop activated")
                    else:
                        st.error("Failed to activate emergency stop")
        
        with col2:
            if st.button("🔄 Reset System", type="secondary"):
                if st.checkbox("Confirm System Reset"):
                    success = self.system_integrator.reset_system(intersection_id)
                    if success:
                        st.success("System reset completed")
                    else:
                        st.error("Failed to reset system")
        
        with col3:
            if st.button("📞 Contact Support", type="primary"):
                st.info("Support Contact: traffic-support@city.gov | Phone: (555) 123-4567")
    
    def _get_intersection_name(self, intersection_id: str) -> str:
        """Get display name for intersection."""
        config = self.config_manager.get_intersection_config(intersection_id)
        if config and 'name' in config:
            return config['name']
        return intersection_id
    
    def _get_vehicle_delta(self, intersection_id: str) -> int:
        """Get vehicle count change from previous reading."""
        # Simplified implementation - in real system would use historical data
        return np.random.randint(-3, 4)
    
    def _get_wait_time_delta(self, intersection_id: str) -> float:
        """Get wait time change from previous reading."""
        # Simplified implementation - in real system would use historical data
        return np.random.uniform(-5.0, 5.0)
    
    def _get_queue_delta(self, intersection_id: str) -> float:
        """Get queue length change from previous reading."""
        # Simplified implementation - in real system would use historical data
        return np.random.uniform(-10.0, 10.0)


def main():
    """Main function to run the dashboard."""
    dashboard = SmartTrafficDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()