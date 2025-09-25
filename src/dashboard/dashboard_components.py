"""Dashboard components for the Smart Traffic Management System."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

try:
    from ..models.traffic_state import TrafficState
    from ..models.signal_action import SignalAction
    from ..agents.signal_control_manager import SignalControlManager, IntersectionSignalState
    from ..processors.traffic_aggregator import TrafficAggregator
    from ..agents.q_learning_agent import QLearningAgent
    from ..processors.prediction_engine import PredictionEngine, PredictionResult
    from ..config.config_manager import ConfigManager
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
    from processors.prediction_engine import PredictionEngine, PredictionResult
    from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class TrafficMonitor:
    """Component for real-time traffic monitoring displays."""
    
    def __init__(self):
        self.update_interval = 5  # seconds
    
    def render_vehicle_counts_chart(self, traffic_state: TrafficState) -> None:
        """Render vehicle counts by direction chart."""
        directions = list(traffic_state.vehicle_counts.keys())
        counts = list(traffic_state.vehicle_counts.values())
        
        fig = px.bar(
            x=directions, y=counts,
            title="Vehicle Counts by Direction",
            labels={'x': 'Direction', 'y': 'Vehicle Count'},
            color=counts,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_queue_analysis(self, traffic_state: TrafficState) -> None:
        """Render queue length analysis."""
        directions = list(traffic_state.queue_lengths.keys())
        queue_lengths = list(traffic_state.queue_lengths.values())
        wait_times = [traffic_state.wait_times.get(d, 0) for d in directions]
        
        # Create subplot with queue lengths and wait times
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Queue Length (m)',
            x=directions,
            y=queue_lengths,
            yaxis='y',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            name='Wait Time (s)',
            x=directions,
            y=wait_times,
            yaxis='y2',
            mode='lines+markers',
            marker_color='red',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title='Queue Length vs Wait Time Analysis',
            xaxis=dict(title='Direction'),
            yaxis=dict(title='Queue Length (meters)', side='left'),
            yaxis2=dict(title='Wait Time (seconds)', side='right', overlaying='y'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_status(self, signal_state: IntersectionSignalState) -> None:
        """Render current signal status display."""
        st.subheader("Current Signal Status")
        
        # Create columns for each phase
        phases = list(signal_state.phases.keys())
        cols = st.columns(len(phases))
        
        for i, (phase_id, phase_state) in enumerate(signal_state.phases.items()):
            with cols[i]:
                phase_name = phase_id.replace('_', ' ').title()
                
                # Signal state indicator
                if phase_state.current_state.value == 'green':
                    st.markdown(f'<div style="text-align: center; color: green; font-size: 24px;">🟢</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center; font-weight: bold;">{phase_name}</div>', 
                               unsafe_allow_html=True)
                    
                    # Progress bar
                    progress = phase_state.get_progress()
                    st.progress(progress)
                    st.write(f"Remaining: {phase_state.time_remaining}s")
                    
                elif phase_state.current_state.value == 'yellow':
                    st.markdown(f'<div style="text-align: center; color: orange; font-size: 24px;">🟡</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center; font-weight: bold;">{phase_name}</div>', 
                               unsafe_allow_html=True)
                    st.write(f"Remaining: {phase_state.time_remaining}s")
                    
                else:  # red
                    st.markdown(f'<div style="text-align: center; color: red; font-size: 24px;">🔴</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center; font-weight: bold;">{phase_name}</div>', 
                               unsafe_allow_html=True)
        
        # Manual override indicator
        if signal_state.manual_override:
            st.warning(f"🚨 Manual Override Active - Operator: {signal_state.override_operator}")
    
    def render_congestion_heatmap(self, traffic_state: TrafficState) -> None:
        """Render congestion level heatmap."""
        directions = list(traffic_state.vehicle_counts.keys())
        
        # Calculate congestion score for each direction
        congestion_scores = []
        for direction in directions:
            vehicle_count = traffic_state.vehicle_counts.get(direction, 0)
            queue_length = traffic_state.queue_lengths.get(direction, 0)
            wait_time = traffic_state.wait_times.get(direction, 0)
            
            # Normalize and combine metrics (0-100 scale)
            vehicle_score = min(100, vehicle_count * 5)  # 20 vehicles = 100
            queue_score = min(100, queue_length * 2)     # 50m queue = 100
            wait_score = min(100, wait_time / 1.2)       # 120s wait = 100
            
            congestion_score = (vehicle_score + queue_score + wait_score) / 3
            congestion_scores.append(congestion_score)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[congestion_scores],
            x=directions,
            y=['Congestion Level'],
            colorscale='RdYlGn_r',  # Red-Yellow-Green reversed
            zmin=0,
            zmax=100,
            colorbar=dict(title="Congestion Score")
        ))
        
        fig.update_layout(
            title="Congestion Heatmap by Direction",
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)


class PerformanceMetrics:
    """Component for displaying performance metrics and KPIs."""
    
    def __init__(self):
        self.baseline_metrics = {
            'average_wait_time': 75.0,  # seconds
            'throughput': 180.0,        # vehicles per hour
            'queue_clearance_time': 90.0  # seconds
        }
    
    def render_kpi_dashboard(self, current_metrics: Dict[str, float]) -> None:
        """Render KPI dashboard with current vs baseline comparison."""
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_throughput = current_metrics.get('throughput', 0)
            baseline_throughput = self.baseline_metrics['throughput']
            improvement = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
            
            st.metric(
                "Throughput",
                f"{current_throughput:.1f} veh/hr",
                delta=f"{improvement:+.1f}%"
            )
        
        with col2:
            current_wait = current_metrics.get('average_delay', 0)
            baseline_wait = self.baseline_metrics['average_wait_time']
            improvement = ((baseline_wait - current_wait) / baseline_wait) * 100
            
            st.metric(
                "Avg Wait Time",
                f"{current_wait:.1f}s",
                delta=f"{improvement:+.1f}%"
            )
        
        with col3:
            current_efficiency = current_metrics.get('efficiency', 0) * 100
            st.metric(
                "System Efficiency",
                f"{current_efficiency:.1f}%"
            )
        
        with col4:
            current_clearance = current_metrics.get('queue_clearance_time', 0)
            baseline_clearance = self.baseline_metrics['queue_clearance_time']
            improvement = ((baseline_clearance - current_clearance) / baseline_clearance) * 100
            
            st.metric(
                "Queue Clearance",
                f"{current_clearance:.1f}s",
                delta=f"{improvement:+.1f}%"
            )
    
    def render_commute_time_analysis(self, current_metrics: Dict[str, float]) -> None:
        """Render commute time reduction analysis."""
        st.subheader("Commute Time Impact Analysis")
        
        current_wait = current_metrics.get('average_delay', 0)
        baseline_wait = self.baseline_metrics['average_wait_time']
        
        # Calculate reduction percentage
        if baseline_wait > 0:
            reduction_pct = ((baseline_wait - current_wait) / baseline_wait) * 100
        else:
            reduction_pct = 0
        
        # Display progress toward 10% target
        target_reduction = 10.0
        progress = min(1.0, reduction_pct / target_reduction)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Progress bar toward target
            st.write("**Progress Toward 10% Reduction Target**")
            st.progress(progress)
            
            if reduction_pct >= target_reduction:
                st.success(f"🎯 Target Achieved! {reduction_pct:.1f}% reduction")
            elif reduction_pct >= target_reduction * 0.7:
                st.info(f"📈 Good Progress: {reduction_pct:.1f}% reduction")
            else:
                st.warning(f"⚠️ Below Target: {reduction_pct:.1f}% reduction")
        
        with col2:
            # Numeric display
            st.metric(
                "Commute Time Reduction",
                f"{reduction_pct:.1f}%",
                delta=f"{reduction_pct - target_reduction:.1f}% vs target"
            )
    
    def render_before_after_comparison(self, current_metrics: Dict[str, float]) -> None:
        """Render before/after comparison charts."""
        st.subheader("Before/After System Implementation")
        
        metrics = ['Wait Time (s)', 'Queue Length (m)', 'Throughput (veh/hr)']
        before_values = [75, 45, 180]
        after_values = [
            current_metrics.get('average_delay', 60),
            current_metrics.get('average_queue_length', 35),
            current_metrics.get('throughput', 200)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Before AI System',
            x=metrics,
            y=before_values,
            marker_color='lightcoral',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='After AI System',
            x=metrics,
            y=after_values,
            marker_color='lightblue',
            opacity=0.8
        ))
        
        fig.update_layout(
            title="Performance Comparison: Before vs After AI Implementation",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


class ManualOverride:
    """Component for manual override controls and emergency intervention."""
    
    def __init__(self):
        self.override_timeout = 300  # 5 minutes default timeout
    
    def render_override_controls(self, intersection_id: str, signal_phases: List[str]) -> Dict[str, Any]:
        """Render manual override control interface."""
        st.subheader("Manual Override Controls")
        
        override_actions = {}
        
        # Override activation
        if not st.session_state.get('manual_override_active', False):
            col1, col2 = st.columns(2)
            
            with col1:
                operator_id = st.text_input("Operator ID", key="manual_operator_id")
            
            with col2:
                reason = st.selectbox(
                    "Override Reason",
                    ["Emergency", "Maintenance", "Special Event", "Testing", "Other"]
                )
            
            if st.button("🚨 Activate Manual Override", type="primary"):
                if operator_id:
                    override_actions['activate'] = {
                        'operator_id': operator_id,
                        'reason': reason,
                        'intersection_id': intersection_id
                    }
                else:
                    st.error("Please enter Operator ID")
        
        else:
            # Override is active - show controls
            st.success(f"Manual Override Active - Operator: {st.session_state.get('override_operator', 'Unknown')}")
            
            # Phase control
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_phase = st.selectbox("Select Phase", signal_phases)
            
            with col2:
                duration = st.slider("Duration (seconds)", 10, 300, 60)
            
            with col3:
                if st.button("Set Phase"):
                    override_actions['set_phase'] = {
                        'phase': selected_phase,
                        'duration': duration,
                        'intersection_id': intersection_id
                    }
            
            # Emergency controls
            st.markdown("---")
            st.subheader("Emergency Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚨 All Red", type="secondary"):
                    override_actions['all_red'] = {'intersection_id': intersection_id}
            
            with col2:
                if st.button("🚑 Emergency Vehicle Priority", type="secondary"):
                    priority_direction = st.selectbox("Priority Direction", signal_phases)
                    override_actions['emergency_priority'] = {
                        'direction': priority_direction,
                        'intersection_id': intersection_id
                    }
            
            with col3:
                if st.button("❌ Deactivate Override"):
                    override_actions['deactivate'] = {'intersection_id': intersection_id}
        
        return override_actions
    
    def render_override_history(self, override_history: List[Dict]) -> None:
        """Render history of manual override actions."""
        st.subheader("Override History")
        
        if override_history:
            df = pd.DataFrame(override_history)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No override history available")
    
    def render_emergency_procedures(self) -> None:
        """Render emergency procedures and contact information."""
        with st.expander("🚨 Emergency Procedures"):
            st.markdown("""
            ### Emergency Response Procedures
            
            **Level 1 - Traffic Congestion:**
            1. Monitor traffic conditions
            2. Consider manual phase adjustments
            3. Document actions taken
            
            **Level 2 - System Malfunction:**
            1. Activate manual override
            2. Set safe signal timing
            3. Contact technical support
            4. Monitor until resolved
            
            **Level 3 - Emergency Vehicles:**
            1. Activate emergency vehicle priority
            2. Clear path for emergency vehicles
            3. Resume normal operation after passage
            
            **Level 4 - Critical Emergency:**
            1. Set all signals to red
            2. Contact emergency services
            3. Direct traffic manually if safe
            4. Wait for emergency response
            
            ### Emergency Contacts
            - **Technical Support:** (555) 123-4567
            - **Emergency Services:** 911
            - **Traffic Control Center:** (555) 987-6543
            """)


class AnalyticsSection:
    """Component for analytics, trends, and reporting."""
    
    def __init__(self):
        self.analysis_periods = {
            "Last Hour": 1,
            "Last 4 Hours": 4,
            "Last 24 Hours": 24,
            "Last Week": 168
        }
    
    def render_traffic_trends(self, trend_data: Dict[str, Any], time_period: str) -> None:
        """Render traffic trend analysis."""
        st.subheader(f"Traffic Trends - {time_period}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trend_direction = trend_data.get('trend_direction', 'stable')
            trend_icons = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
            st.metric("Traffic Trend", f"{trend_icons[trend_direction]} {trend_direction.title()}")
        
        with col2:
            avg_vehicles = trend_data.get('average_vehicles', 0)
            st.metric("Average Vehicles", f"{avg_vehicles:.1f}")
        
        with col3:
            peak_vehicles = trend_data.get('peak_vehicles', 0)
            st.metric("Peak Vehicles", peak_vehicles)
        
        with col4:
            congestion_level = trend_data.get('congestion_level', 'low')
            congestion_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.metric("Congestion Level", f"{congestion_colors[congestion_level]} {congestion_level.title()}")
    
    def render_prediction_analysis(self, predictions: Optional[PredictionResult]) -> None:
        """Render traffic prediction analysis."""
        st.subheader("Traffic Predictions")
        
        if predictions and predictions.predictions:
            # Create prediction chart
            pred_df = pd.DataFrame({
                'Time': predictions.timestamps,
                'Predicted Vehicles': predictions.predictions
            })
            
            fig = px.line(
                pred_df, x='Time', y='Predicted Vehicles',
                title=f"Traffic Predictions - {predictions.horizon_minutes} minute horizon",
                markers=True
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction confidence
            confidence_pct = predictions.confidence * 100
            if confidence_pct >= 80:
                st.success(f"High Confidence: {confidence_pct:.1f}%")
            elif confidence_pct >= 60:
                st.info(f"Medium Confidence: {confidence_pct:.1f}%")
            else:
                st.warning(f"Low Confidence: {confidence_pct:.1f}%")
            
            # Features used
            st.write(f"**Features Used:** {', '.join(predictions.features_used)}")
        
        else:
            st.info("No predictions available")
    
    def render_optimization_impact(self, optimization_data: List[Dict]) -> None:
        """Render signal optimization impact analysis."""
        st.subheader("Signal Optimization Impact")
        
        if optimization_data:
            df = pd.DataFrame(optimization_data)
            
            # Before/after comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Before Optimization',
                x=df['timestamp'],
                y=df['wait_time_before'],
                marker_color='lightcoral',
                opacity=0.7
            ))
            
            fig.add_trace(go.Bar(
                name='After Optimization',
                x=df['timestamp'],
                y=df['wait_time_after'],
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Wait Time Before/After Optimization Events",
                xaxis_title="Time",
                yaxis_title="Average Wait Time (seconds)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate average improvement
            avg_before = df['wait_time_before'].mean()
            avg_after = df['wait_time_after'].mean()
            improvement = ((avg_before - avg_after) / avg_before) * 100
            
            st.metric("Average Improvement", f"{improvement:.1f}%")
        
        else:
            st.info("No optimization data available")
    
    def render_export_options(self, intersection_id: str) -> Dict[str, Any]:
        """Render data export options."""
        st.subheader("Data Export")
        
        export_actions = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_period = st.selectbox(
                "Export Period",
                list(self.analysis_periods.keys())
            )
        
        with col2:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Excel"]
            )
        
        with col3:
            export_type = st.selectbox(
                "Data Type",
                ["Traffic Data", "Performance Metrics", "System Config", "All Data"]
            )
        
        if st.button("📊 Generate Export"):
            export_actions['generate'] = {
                'intersection_id': intersection_id,
                'period': export_period,
                'format': export_format,
                'type': export_type,
                'hours': self.analysis_periods[export_period]
            }
        
        return export_actions


class SystemIntegrator:
    """Integrates all system components for the dashboard."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.system_components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize components for each intersection
            for intersection_id in self.config_manager.get_all_intersection_ids():
                intersection_config_data = self.config_manager.get_intersection_config(intersection_id)
                
                if intersection_config_data:
                    # Create intersection config object
                    try:
                        from ..config.intersection_config import IntersectionConfig
                    except ImportError:
                        from config.intersection_config import IntersectionConfig
                    intersection_config = IntersectionConfig(
                        intersection_id=intersection_id,
                        **intersection_config_data
                    )
                    
                    # Initialize components
                    signal_manager = SignalControlManager(intersection_config)
                    traffic_aggregator = TrafficAggregator(intersection_id, intersection_config_data)
                    
                    # Initialize RL agent
                    signal_phases = intersection_config_data['geometry']['signal_phases']
                    rl_agent = QLearningAgent(intersection_id, signal_phases)
                    
                    # Initialize prediction engine
                    prediction_engine = PredictionEngine()
                    
                    self.system_components[intersection_id] = {
                        'signal_manager': signal_manager,
                        'traffic_aggregator': traffic_aggregator,
                        'rl_agent': rl_agent,
                        'prediction_engine': prediction_engine,
                        'intersection_config': intersection_config
                    }
                    
                    logger.info(f"Initialized components for intersection {intersection_id}")
        
        except Exception as e:
            logger.error(f"Error initializing system components: {e}")
    
    def get_system_status(self, intersection_id: str) -> Dict[str, Any]:
        """Get overall system status."""
        if intersection_id not in self.system_components:
            return {
                'overall_status': 'error',
                'components': {},
                'message': 'Intersection not found'
            }
        
        components = self.system_components[intersection_id]
        component_status = {}
        
        # Check each component
        try:
            # Signal manager
            signal_state = components['signal_manager'].get_current_signal_state()
            component_status['signal_control'] = 'active' if signal_state else 'inactive'
            
            # Traffic aggregator
            component_status['traffic_monitoring'] = 'active'
            
            # RL agent
            rl_stats = components['rl_agent'].get_training_stats()
            component_status['rl_optimization'] = 'active' if rl_stats['episodes'] > 0 else 'inactive'
            
            # Prediction engine
            component_status['traffic_prediction'] = 'active' if components['prediction_engine'].is_trained else 'inactive'
            
            # Determine overall status
            active_components = sum(1 for status in component_status.values() if status == 'active')
            total_components = len(component_status)
            
            if active_components == total_components:
                overall_status = 'healthy'
            elif active_components >= total_components * 0.7:
                overall_status = 'warning'
            else:
                overall_status = 'error'
            
            return {
                'overall_status': overall_status,
                'components': component_status,
                'active_components': active_components,
                'total_components': total_components
            }
        
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'overall_status': 'error',
                'components': component_status,
                'message': str(e)
            }
    
    def get_current_traffic_state(self, intersection_id: str) -> TrafficState:
        """Get current traffic state for intersection."""
        if intersection_id not in self.system_components:
            # Return empty state
            return TrafficState(
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                vehicle_counts={'north': 0, 'south': 0, 'east': 0, 'west': 0},
                queue_lengths={'north': 0, 'south': 0, 'east': 0, 'west': 0},
                wait_times={'north': 0, 'south': 0, 'east': 0, 'west': 0},
                signal_phase='north_south_green',
                prediction_confidence=0.0
            )
        
        # Generate simulated current traffic state
        # In real implementation, this would come from live data
        directions = ['north', 'south', 'east', 'west']
        
        # Simulate realistic traffic patterns based on time of day
        current_hour = datetime.now().hour
        base_traffic = 10
        
        if current_hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
            base_traffic = 25
        elif current_hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night hours
            base_traffic = 5
        
        vehicle_counts = {
            direction: max(0, int(base_traffic + np.random.normal(0, 5)))
            for direction in directions
        }
        
        queue_lengths = {
            direction: max(0, count * 2.5 + np.random.normal(0, 5))
            for direction, count in vehicle_counts.items()
        }
        
        wait_times = {
            direction: max(0, count * 3 + np.random.normal(0, 10))
            for direction, count in vehicle_counts.items()
        }
        
        return TrafficState(
            intersection_id=intersection_id,
            timestamp=datetime.now(),
            vehicle_counts=vehicle_counts,
            queue_lengths=queue_lengths,
            wait_times=wait_times,
            signal_phase='north_south_green',
            prediction_confidence=0.75 + np.random.normal(0, 0.1)
        )
    
    def get_current_signal_state(self, intersection_id: str) -> IntersectionSignalState:
        """Get current signal state for intersection."""
        if intersection_id not in self.system_components:
            # Return default state
            from ..agents.signal_control_manager import SignalState, PhaseState
            
            phases = {
                'north_south_green': PhaseState('north_south_green', SignalState.GREEN, 30, 45),
                'east_west_green': PhaseState('east_west_green', SignalState.RED, 0, 0)
            }
            
            return IntersectionSignalState(
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                phases=phases,
                current_cycle_phase='north_south_green',
                cycle_start_time=datetime.now()
            )
        
        components = self.system_components[intersection_id]
        return components['signal_manager'].get_current_signal_state()
    
    def get_performance_metrics(self, intersection_id: str) -> Dict[str, float]:
        """Get performance metrics for intersection."""
        if intersection_id not in self.system_components:
            return {
                'throughput': 0.0,
                'efficiency': 0.0,
                'average_delay': 0.0,
                'queue_clearance_time': 0.0
            }
        
        components = self.system_components[intersection_id]
        return components['traffic_aggregator'].get_performance_metrics()
    
    def get_traffic_trends(self, intersection_id: str, hours: int = 1) -> Dict[str, Any]:
        """Get traffic trends for intersection."""
        if intersection_id not in self.system_components:
            return {
                'trend_direction': 'stable',
                'average_vehicles': 0,
                'peak_vehicles': 0,
                'average_wait_time': 0.0,
                'congestion_level': 'low'
            }
        
        components = self.system_components[intersection_id]
        return components['traffic_aggregator'].get_traffic_trends(hours)
    
    def enable_manual_override(self, intersection_id: str, operator_id: str) -> bool:
        """Enable manual override for intersection."""
        if intersection_id not in self.system_components:
            return False
        
        components = self.system_components[intersection_id]
        return components['signal_manager'].enable_manual_override(operator_id)
    
    def disable_manual_override(self, intersection_id: str) -> bool:
        """Disable manual override for intersection."""
        if intersection_id not in self.system_components:
            return False
        
        components = self.system_components[intersection_id]
        return components['signal_manager'].disable_manual_override()
    
    def set_manual_phase(self, intersection_id: str, phase_id: str, duration: int) -> bool:
        """Set manual phase for intersection."""
        if intersection_id not in self.system_components:
            return False
        
        components = self.system_components[intersection_id]
        return components['signal_manager'].set_manual_phase(phase_id, duration)
    
    def get_recent_actions(self, intersection_id: str, limit: int = 10) -> List:
        """Get recent signal actions for intersection."""
        if intersection_id not in self.system_components:
            return []
        
        components = self.system_components[intersection_id]
        return components['signal_manager'].get_action_history(limit)
    
    def get_rl_training_stats(self, intersection_id: str) -> Dict[str, Any]:
        """Get RL agent training statistics."""
        if intersection_id not in self.system_components:
            return {'episodes': 0, 'average_reward': 0.0, 'epsilon': 0.0, 'q_table_size': 0}
        
        components = self.system_components[intersection_id]
        return components['rl_agent'].get_training_stats()
    
    def get_prediction_status(self, intersection_id: str) -> Dict[str, Any]:
        """Get prediction engine status."""
        if intersection_id not in self.system_components:
            return {'status': 'inactive', 'confidence': 0.0, 'last_training': 'Never'}
        
        components = self.system_components[intersection_id]
        prediction_engine = components['prediction_engine']
        
        return {
            'status': 'active' if prediction_engine.is_trained else 'inactive',
            'confidence': 0.75,  # Simplified
            'last_training': 'Recently'  # Simplified
        }
    
    def generate_traffic_predictions(self, intersection_id: str) -> Optional[PredictionResult]:
        """Generate traffic predictions for intersection."""
        if intersection_id not in self.system_components:
            return None
        
        try:
            components = self.system_components[intersection_id]
            prediction_engine = components['prediction_engine']
            
            # Generate sample recent states for prediction
            recent_states = []
            for i in range(12):  # 12 time steps
                timestamp = datetime.now() - timedelta(minutes=i*5)
                state = self.get_current_traffic_state(intersection_id)
                state.timestamp = timestamp
                recent_states.append(state)
            
            recent_states.reverse()  # Chronological order
            
            if prediction_engine.is_trained:
                return prediction_engine.predict_traffic_volume(intersection_id, recent_states)
            else:
                return prediction_engine.get_fallback_prediction(intersection_id, recent_states)
        
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None
    
    def export_traffic_data(self, intersection_id: str, hours: int = 24) -> List[Dict]:
        """Export traffic data for intersection."""
        if intersection_id not in self.system_components:
            return []
        
        components = self.system_components[intersection_id]
        return components['traffic_aggregator'].export_traffic_data(hours)
    
    def trigger_training_episode(self, intersection_id: str) -> bool:
        """Trigger RL training episode."""
        if intersection_id not in self.system_components:
            return False
        
        try:
            # Generate sample training data
            traffic_states = []
            for i in range(10):
                state = self.get_current_traffic_state(intersection_id)
                traffic_states.append(state)
            
            components = self.system_components[intersection_id]
            components['rl_agent'].train_episode(traffic_states)
            return True
        
        except Exception as e:
            logger.error(f"Error triggering training episode: {e}")
            return False
    
    def save_rl_model(self, intersection_id: str) -> bool:
        """Save RL model for intersection."""
        if intersection_id not in self.system_components:
            return False
        
        try:
            components = self.system_components[intersection_id]
            model_path = f"models/rl_model_{intersection_id}.pkl"
            components['rl_agent'].save_model(model_path)
            return True
        
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
            return False
    
    def retrain_prediction_model(self, intersection_id: str) -> bool:
        """Retrain prediction model for intersection."""
        if intersection_id not in self.system_components:
            return False
        
        try:
            # Generate sample training data
            traffic_states = []
            for i in range(100):  # 100 historical states
                state = self.get_current_traffic_state(intersection_id)
                traffic_states.append(state)
            
            components = self.system_components[intersection_id]
            components['prediction_engine'].train_model(traffic_states, epochs=50)
            return True
        
        except Exception as e:
            logger.error(f"Error retraining prediction model: {e}")
            return False
    
    def emergency_stop(self, intersection_id: str) -> bool:
        """Emergency stop for intersection."""
        if intersection_id not in self.system_components:
            return False
        
        try:
            components = self.system_components[intersection_id]
            # Set all phases to red (emergency stop)
            return components['signal_manager'].set_manual_phase('all_red', 300)
        
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")
            return False
    
    def reset_system(self, intersection_id: str) -> bool:
        """Reset system for intersection."""
        if intersection_id not in self.system_components:
            return False
        
        try:
            components = self.system_components[intersection_id]
            
            # Reset signal manager
            components['signal_manager'].reset_to_defaults()
            components['signal_manager'].disable_manual_override()
            
            # Reset RL agent exploration rate
            components['rl_agent'].config.epsilon = 0.1
            
            logger.info(f"System reset completed for intersection {intersection_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            return False