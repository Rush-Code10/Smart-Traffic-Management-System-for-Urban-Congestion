"""Tests for the Smart Traffic Management System Dashboard."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# Import dashboard components
from src.dashboard.dashboard_components import (
    TrafficMonitor, PerformanceMetrics, ManualOverride, 
    AnalyticsSection, SystemIntegrator
)
from src.dashboard.main_dashboard import SmartTrafficDashboard
from src.models.traffic_state import TrafficState
from src.models.signal_action import SignalAction
from src.agents.signal_control_manager import IntersectionSignalState, PhaseState, SignalState
from src.processors.prediction_engine import PredictionResult
from src.config.config_manager import ConfigManager


class TestTrafficMonitor:
    """Test cases for TrafficMonitor component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.traffic_monitor = TrafficMonitor()
        self.sample_traffic_state = TrafficState(
            intersection_id="test_001",
            timestamp=datetime.now(),
            vehicle_counts={'north': 15, 'south': 12, 'east': 8, 'west': 10},
            queue_lengths={'north': 25.0, 'south': 20.0, 'east': 15.0, 'west': 18.0},
            wait_times={'north': 45.0, 'south': 38.0, 'east': 30.0, 'west': 35.0},
            signal_phase='north_south_green',
            prediction_confidence=0.85
        )
    
    def test_initialization(self):
        """Test TrafficMonitor initialization."""
        assert self.traffic_monitor.update_interval == 5
    
    @patch('streamlit.plotly_chart')
    def test_render_vehicle_counts_chart(self, mock_plotly_chart):
        """Test vehicle counts chart rendering."""
        self.traffic_monitor.render_vehicle_counts_chart(self.sample_traffic_state)
        mock_plotly_chart.assert_called_once()
    
    @patch('streamlit.plotly_chart')
    def test_render_queue_analysis(self, mock_plotly_chart):
        """Test queue analysis rendering."""
        self.traffic_monitor.render_queue_analysis(self.sample_traffic_state)
        mock_plotly_chart.assert_called_once()
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    def test_render_signal_status(self, mock_columns, mock_subheader):
        """Test signal status rendering."""
        # Create mock signal state
        phases = {
            'north_south_green': PhaseState('north_south_green', SignalState.GREEN, 30, 45),
            'east_west_green': PhaseState('east_west_green', SignalState.RED, 0, 45)
        }
        
        signal_state = IntersectionSignalState(
            intersection_id="test_001",
            timestamp=datetime.now(),
            phases=phases,
            current_cycle_phase='north_south_green',
            cycle_start_time=datetime.now()
        )
        
        # Mock columns
        mock_columns.return_value = [Mock(), Mock()]
        
        self.traffic_monitor.render_signal_status(signal_state)
        mock_subheader.assert_called_with("Current Signal Status")
    
    @patch('streamlit.plotly_chart')
    def test_render_congestion_heatmap(self, mock_plotly_chart):
        """Test congestion heatmap rendering."""
        self.traffic_monitor.render_congestion_heatmap(self.sample_traffic_state)
        mock_plotly_chart.assert_called_once()


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.performance_metrics = PerformanceMetrics()
        self.sample_metrics = {
            'throughput': 200.0,
            'efficiency': 0.85,
            'average_delay': 45.0,
            'queue_clearance_time': 60.0
        }
    
    def test_initialization(self):
        """Test PerformanceMetrics initialization."""
        assert 'average_wait_time' in self.performance_metrics.baseline_metrics
        assert 'throughput' in self.performance_metrics.baseline_metrics
        assert 'queue_clearance_time' in self.performance_metrics.baseline_metrics
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.subheader')
    def test_render_kpi_dashboard(self, mock_subheader, mock_metric, mock_columns):
        """Test KPI dashboard rendering."""
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        self.performance_metrics.render_kpi_dashboard(self.sample_metrics)
        mock_subheader.assert_called_with("Key Performance Indicators")
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.success')
    def test_render_commute_time_analysis_target_achieved(self, mock_success, mock_metric, mock_columns):
        """Test commute time analysis when target is achieved."""
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        
        # Set metrics that achieve 10% reduction target
        metrics_with_good_reduction = {
            'average_delay': 60.0  # Baseline is 75.0, so this is 20% reduction
        }
        
        self.performance_metrics.render_commute_time_analysis(metrics_with_good_reduction)
        mock_success.assert_called()
    
    @patch('streamlit.plotly_chart')
    def test_render_before_after_comparison(self, mock_plotly_chart):
        """Test before/after comparison chart."""
        self.performance_metrics.render_before_after_comparison(self.sample_metrics)
        mock_plotly_chart.assert_called_once()


class TestManualOverride:
    """Test cases for ManualOverride component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manual_override = ManualOverride()
        self.signal_phases = ['north_south_green', 'east_west_green']
    
    def test_initialization(self):
        """Test ManualOverride initialization."""
        assert self.manual_override.override_timeout == 300
    
    @patch('streamlit.session_state', {'manual_override_active': False})
    @patch('streamlit.text_input')
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    @patch('streamlit.columns')
    def test_render_override_controls_inactive(self, mock_columns, mock_button, 
                                             mock_selectbox, mock_text_input):
        """Test override controls when override is inactive."""
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "operator123"
        mock_selectbox.return_value = "Emergency"
        mock_button.return_value = True
        
        actions = self.manual_override.render_override_controls("test_001", self.signal_phases)
        
        # Should have activate action when button is pressed with operator ID
        assert 'activate' in actions
        assert actions['activate']['operator_id'] == "operator123"
    
    @patch('streamlit.session_state', {'manual_override_active': True, 'override_operator': 'test_op'})
    @patch('streamlit.selectbox')
    @patch('streamlit.slider')
    @patch('streamlit.button')
    @patch('streamlit.columns')
    def test_render_override_controls_active(self, mock_columns, mock_button, 
                                           mock_slider, mock_selectbox):
        """Test override controls when override is active."""
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        mock_selectbox.return_value = "north_south_green"
        mock_slider.return_value = 60
        mock_button.return_value = True
        
        actions = self.manual_override.render_override_controls("test_001", self.signal_phases)
        
        # Should have set_phase action when button is pressed
        assert 'set_phase' in actions
        assert actions['set_phase']['phase'] == "north_south_green"
        assert actions['set_phase']['duration'] == 60
    
    @patch('streamlit.dataframe')
    def test_render_override_history_with_data(self, mock_dataframe):
        """Test override history rendering with data."""
        history = [
            {'timestamp': datetime.now(), 'operator': 'test_op', 'action': 'activate'},
            {'timestamp': datetime.now(), 'operator': 'test_op', 'action': 'set_phase'}
        ]
        
        self.manual_override.render_override_history(history)
        mock_dataframe.assert_called_once()
    
    @patch('streamlit.info')
    def test_render_override_history_empty(self, mock_info):
        """Test override history rendering with no data."""
        self.manual_override.render_override_history([])
        mock_info.assert_called_with("No override history available")
    
    @patch('streamlit.expander')
    def test_render_emergency_procedures(self, mock_expander):
        """Test emergency procedures rendering."""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        self.manual_override.render_emergency_procedures()
        mock_expander.assert_called_once()


class TestAnalyticsSection:
    """Test cases for AnalyticsSection component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analytics_section = AnalyticsSection()
        self.sample_trend_data = {
            'trend_direction': 'increasing',
            'average_vehicles': 18.5,
            'peak_vehicles': 35,
            'average_wait_time': 42.0,
            'congestion_level': 'medium'
        }
    
    def test_initialization(self):
        """Test AnalyticsSection initialization."""
        assert "Last Hour" in self.analytics_section.analysis_periods
        assert "Last Week" in self.analytics_section.analysis_periods
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_render_traffic_trends(self, mock_metric, mock_columns):
        """Test traffic trends rendering."""
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        self.analytics_section.render_traffic_trends(self.sample_trend_data, "Last Hour")
        
        # Should call metric for each trend indicator
        assert mock_metric.call_count >= 4
    
    @patch('streamlit.plotly_chart')
    def test_render_prediction_analysis_with_data(self, mock_plotly_chart):
        """Test prediction analysis with valid predictions."""
        predictions = PredictionResult(
            intersection_id="test_001",
            predictions=[15.0, 18.0, 22.0, 20.0, 16.0],
            timestamps=[datetime.now() + timedelta(minutes=i*5) for i in range(5)],
            confidence=0.85,
            horizon_minutes=25,
            features_used=['vehicle_count', 'time_of_day']
        )
        
        self.analytics_section.render_prediction_analysis(predictions)
        mock_plotly_chart.assert_called_once()
    
    @patch('streamlit.info')
    def test_render_prediction_analysis_no_data(self, mock_info):
        """Test prediction analysis with no predictions."""
        self.analytics_section.render_prediction_analysis(None)
        mock_info.assert_called_with("No predictions available")
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.metric')
    def test_render_optimization_impact(self, mock_metric, mock_plotly_chart):
        """Test optimization impact rendering."""
        optimization_data = [
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'wait_time_before': 75.0,
                'wait_time_after': 60.0
            },
            {
                'timestamp': datetime.now(),
                'wait_time_before': 80.0,
                'wait_time_after': 65.0
            }
        ]
        
        self.analytics_section.render_optimization_impact(optimization_data)
        mock_plotly_chart.assert_called_once()
        mock_metric.assert_called_once()
    
    @patch('streamlit.columns')
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    def test_render_export_options(self, mock_button, mock_selectbox, mock_columns):
        """Test export options rendering."""
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        mock_selectbox.side_effect = ["Last Hour", "CSV", "Traffic Data"]
        mock_button.return_value = True
        
        actions = self.analytics_section.render_export_options("test_001")
        
        assert 'generate' in actions
        assert actions['generate']['intersection_id'] == "test_001"
        assert actions['generate']['format'] == "CSV"


class TestSystemIntegrator:
    """Test cases for SystemIntegrator component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config_manager = Mock(spec=ConfigManager)
        self.mock_config_manager.get_all_intersection_ids.return_value = ["test_001"]
        self.mock_config_manager.get_intersection_config.return_value = {
            'name': 'Test Intersection',
            'geometry': {
                'lanes': {'north': {'count': 2}, 'south': {'count': 2}},
                'signal_phases': ['north_south_green', 'east_west_green'],
                'default_phase_timings': {'north_south_green': 45, 'east_west_green': 45}
            },
            'camera_positions': {}
        }
        
        self.system_integrator = SystemIntegrator(self.mock_config_manager)
    
    def test_initialization(self):
        """Test SystemIntegrator initialization."""
        assert self.system_integrator.config_manager == self.mock_config_manager
        assert hasattr(self.system_integrator, 'system_components')
    
    def test_get_system_status_healthy(self):
        """Test system status when all components are healthy."""
        status = self.system_integrator.get_system_status("test_001")
        
        assert 'overall_status' in status
        assert 'components' in status
        assert status['overall_status'] in ['healthy', 'warning', 'error']
    
    def test_get_system_status_intersection_not_found(self):
        """Test system status for non-existent intersection."""
        status = self.system_integrator.get_system_status("nonexistent")
        
        assert status['overall_status'] == 'error'
        assert 'message' in status
        assert status['message'] == 'Intersection not found'
    
    def test_get_current_traffic_state(self):
        """Test getting current traffic state."""
        traffic_state = self.system_integrator.get_current_traffic_state("test_001")
        
        assert isinstance(traffic_state, TrafficState)
        assert traffic_state.intersection_id == "test_001"
        assert 'north' in traffic_state.vehicle_counts
        assert 'south' in traffic_state.vehicle_counts
    
    def test_get_current_signal_state(self):
        """Test getting current signal state."""
        signal_state = self.system_integrator.get_current_signal_state("test_001")
        
        assert isinstance(signal_state, IntersectionSignalState)
        assert signal_state.intersection_id == "test_001"
        assert len(signal_state.phases) > 0
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        metrics = self.system_integrator.get_performance_metrics("test_001")
        
        assert isinstance(metrics, dict)
        assert 'throughput' in metrics
        assert 'efficiency' in metrics
        assert 'average_delay' in metrics
        assert 'queue_clearance_time' in metrics
    
    def test_get_traffic_trends(self):
        """Test getting traffic trends."""
        trends = self.system_integrator.get_traffic_trends("test_001", hours=1)
        
        assert isinstance(trends, dict)
        assert 'trend_direction' in trends
        assert 'average_vehicles' in trends
        assert 'congestion_level' in trends
    
    def test_enable_manual_override(self):
        """Test enabling manual override."""
        result = self.system_integrator.enable_manual_override("test_001", "operator123")
        
        # Should return boolean result
        assert isinstance(result, bool)
    
    def test_disable_manual_override(self):
        """Test disabling manual override."""
        result = self.system_integrator.disable_manual_override("test_001")
        
        # Should return boolean result
        assert isinstance(result, bool)
    
    def test_generate_traffic_predictions(self):
        """Test generating traffic predictions."""
        predictions = self.system_integrator.generate_traffic_predictions("test_001")
        
        # Should return PredictionResult or None
        assert predictions is None or isinstance(predictions, PredictionResult)
    
    def test_export_traffic_data(self):
        """Test exporting traffic data."""
        data = self.system_integrator.export_traffic_data("test_001", hours=1)
        
        assert isinstance(data, list)
    
    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        result = self.system_integrator.emergency_stop("test_001")
        
        assert isinstance(result, bool)
    
    def test_reset_system(self):
        """Test system reset functionality."""
        result = self.system_integrator.reset_system("test_001")
        
        assert isinstance(result, bool)


class TestSmartTrafficDashboard:
    """Test cases for SmartTrafficDashboard main class."""
    
    @patch('src.dashboard.main_dashboard.ConfigManager')
    @patch('src.dashboard.main_dashboard.SystemIntegrator')
    def setup_method(self, mock_system_integrator, mock_config_manager):
        """Set up test fixtures."""
        self.mock_config_manager = mock_config_manager.return_value
        self.mock_config_manager.get_all_intersection_ids.return_value = ["test_001"]
        
        self.mock_system_integrator = mock_system_integrator.return_value
        
        # Mock streamlit session state
        with patch('streamlit.session_state', {}):
            self.dashboard = SmartTrafficDashboard()
    
    def test_initialization(self):
        """Test SmartTrafficDashboard initialization."""
        assert hasattr(self.dashboard, 'config_manager')
        assert hasattr(self.dashboard, 'system_integrator')
        assert hasattr(self.dashboard, 'intersection_ids')
    
    @patch('streamlit.session_state', {'initialized': False})
    def test_initialize_session_state(self):
        """Test session state initialization."""
        self.dashboard._initialize_session_state()
        
        # Check that session state variables are set
        assert st.session_state.initialized == True
        assert 'selected_intersection' in st.session_state
        assert 'manual_override_active' in st.session_state
    
    def test_get_intersection_name(self):
        """Test getting intersection display name."""
        self.mock_config_manager.get_intersection_config.return_value = {
            'name': 'Main St & Oak Ave'
        }
        
        name = self.dashboard._get_intersection_name("test_001")
        assert name == 'Main St & Oak Ave'
    
    def test_get_intersection_name_fallback(self):
        """Test getting intersection name when config has no name."""
        self.mock_config_manager.get_intersection_config.return_value = {}
        
        name = self.dashboard._get_intersection_name("test_001")
        assert name == "test_001"


class TestDashboardIntegration:
    """Integration tests for dashboard components working together."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_config_manager = Mock(spec=ConfigManager)
        self.mock_config_manager.get_all_intersection_ids.return_value = ["test_001"]
        self.mock_config_manager.get_intersection_config.return_value = {
            'name': 'Test Intersection',
            'geometry': {
                'lanes': {'north': {'count': 2}, 'south': {'count': 2}},
                'signal_phases': ['north_south_green', 'east_west_green'],
                'default_phase_timings': {'north_south_green': 45, 'east_west_green': 45}
            },
            'camera_positions': {}
        }
    
    def test_system_integrator_with_all_components(self):
        """Test SystemIntegrator working with all dashboard components."""
        system_integrator = SystemIntegrator(self.mock_config_manager)
        
        # Test that all component methods work together
        traffic_state = system_integrator.get_current_traffic_state("test_001")
        signal_state = system_integrator.get_current_signal_state("test_001")
        performance_metrics = system_integrator.get_performance_metrics("test_001")
        
        # Verify data consistency
        assert traffic_state.intersection_id == "test_001"
        assert signal_state.intersection_id == "test_001"
        assert isinstance(performance_metrics, dict)
    
    def test_dashboard_components_data_flow(self):
        """Test data flow between dashboard components."""
        system_integrator = SystemIntegrator(self.mock_config_manager)
        traffic_monitor = TrafficMonitor()
        performance_metrics = PerformanceMetrics()
        
        # Get data from system integrator
        traffic_state = system_integrator.get_current_traffic_state("test_001")
        metrics = system_integrator.get_performance_metrics("test_001")
        
        # Verify components can process the data
        assert traffic_state.get_total_vehicles() >= 0
        assert traffic_state.get_average_wait_time() >= 0
        assert metrics['throughput'] >= 0
        assert metrics['efficiency'] >= 0
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.metric')
    def test_end_to_end_dashboard_rendering(self, mock_metric, mock_plotly_chart):
        """Test end-to-end dashboard rendering workflow."""
        system_integrator = SystemIntegrator(self.mock_config_manager)
        traffic_monitor = TrafficMonitor()
        performance_metrics_component = PerformanceMetrics()
        
        # Simulate complete dashboard rendering workflow
        traffic_state = system_integrator.get_current_traffic_state("test_001")
        signal_state = system_integrator.get_current_signal_state("test_001")
        metrics = system_integrator.get_performance_metrics("test_001")
        
        # Render components
        traffic_monitor.render_vehicle_counts_chart(traffic_state)
        traffic_monitor.render_queue_analysis(traffic_state)
        performance_metrics_component.render_kpi_dashboard(metrics)
        
        # Verify rendering calls were made
        assert mock_plotly_chart.call_count >= 2
        assert mock_metric.call_count >= 4


if __name__ == "__main__":
    pytest.main([__file__])