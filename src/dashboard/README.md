# Smart Traffic Management System Dashboard

A comprehensive Streamlit-based dashboard for monitoring and controlling the AI-powered traffic management system.

## Features

### 🔍 Real-time Monitoring
- Live vehicle counts by direction
- Queue length analysis with visual indicators
- Signal phase status with progress indicators
- Congestion heatmaps for quick assessment
- Real-time traffic flow visualization

### 📊 Performance Metrics
- Key Performance Indicators (KPIs) dashboard
- Commute time reduction tracking (target: 10%)
- Before/after system implementation comparisons
- Throughput and efficiency metrics
- Queue clearance time analysis

### 🚨 Manual Override Controls
- Emergency operator intervention capabilities
- Manual signal phase control
- Emergency vehicle priority settings
- Override history and audit trail
- Emergency procedures and contact information

### 📈 Analytics & Trends
- Traffic pattern analysis over multiple time periods
- Prediction accuracy visualization
- Signal optimization impact assessment
- Data export functionality (CSV, JSON, Excel)
- Historical trend analysis

### ⚙️ System Control
- RL agent training controls
- Prediction model management
- System configuration interface
- Emergency stop and reset functions
- Component status monitoring

## Installation

### Prerequisites
```bash
pip install streamlit plotly pandas numpy torch scikit-learn
```

### Quick Start
1. **Run the demo setup:**
   ```bash
   python examples/dashboard_demo.py
   ```

2. **Launch the dashboard:**
   ```bash
   streamlit run src/dashboard/main_dashboard.py
   ```

3. **Or use the launcher script:**
   ```bash
   python src/dashboard/run_dashboard.py
   ```

The dashboard will be available at `http://localhost:8501`

## Dashboard Components

### Main Dashboard (`main_dashboard.py`)
The primary Streamlit application that orchestrates all dashboard components and provides the main user interface.

**Key Features:**
- Intersection selection and switching
- Real-time data updates with configurable refresh rates
- Tabbed interface for different functional areas
- Alert system for traffic conditions
- Session state management

### Dashboard Components (`dashboard_components.py`)
Modular components that handle specific dashboard functionality:

#### TrafficMonitor
- Real-time traffic visualization
- Vehicle count charts
- Queue analysis displays
- Signal status indicators
- Congestion heatmaps

#### PerformanceMetrics
- KPI dashboard rendering
- Commute time impact analysis
- Before/after comparisons
- Performance trend visualization

#### ManualOverride
- Override control interface
- Emergency intervention tools
- Action history tracking
- Emergency procedures display

#### AnalyticsSection
- Traffic trend analysis
- Prediction visualization
- Optimization impact charts
- Data export functionality

#### SystemIntegrator
- Coordinates all system components
- Manages data flow between components
- Handles system status monitoring
- Provides unified API for dashboard operations

## Usage Guide

### 1. Intersection Selection
- Use the sidebar to select which intersection to monitor
- System status indicators show component health
- Auto-refresh controls for real-time updates

### 2. Real-time Monitoring Tab
- **Traffic Flow Visualization:** Live charts showing vehicle counts, queue lengths, and wait times by direction
- **Signal Status:** Current signal phases with progress indicators and remaining time
- **Recent Actions:** History of recent signal timing adjustments with reasoning

### 3. Performance Metrics Tab
- **KPI Overview:** Key metrics with improvement indicators
- **Commute Time Analysis:** Progress toward 10% reduction target
- **Trend Charts:** 24-hour traffic patterns and wait time comparisons

### 4. Analytics & Trends Tab
- **Time Period Selection:** Choose analysis period (hour, day, week)
- **Prediction Analysis:** Traffic forecasts with confidence levels
- **Optimization Impact:** Before/after optimization event analysis
- **Data Export:** Download traffic data, performance reports, or system configs

### 5. System Control Tab
- **RL Agent Control:** Training statistics and model management
- **Prediction Engine:** Model status and retraining options
- **Configuration:** System parameter adjustments
- **Emergency Controls:** Emergency stop and system reset (use with caution)

### 6. Manual Override
Available in the sidebar when needed:
- **Activation:** Enter operator ID and reason for override
- **Phase Control:** Manually set signal phases and durations
- **Emergency Functions:** All-red signals, emergency vehicle priority
- **Deactivation:** Return to automatic control

## Configuration

### System Configuration
The dashboard uses the system configuration from `config/system_config.json`:

```json
{
  "system": {
    "dashboard_refresh_rate_seconds": 5,
    "max_signal_adjustment_seconds": 120,
    "emergency_override_timeout_seconds": 300
  },
  "intersections": {
    "intersection_001": {
      "name": "Main St & Oak Ave",
      "geometry": {
        "signal_phases": ["north_south_green", "east_west_green"],
        "default_phase_timings": {
          "north_south_green": 45,
          "east_west_green": 45
        }
      }
    }
  }
}
```

### Dashboard Customization
Key customization options:

- **Refresh Rate:** Adjust auto-refresh interval (1-30 seconds)
- **Alert Thresholds:** Modify congestion alert levels
- **Chart Colors:** Customize visualization color schemes
- **Metric Targets:** Set performance targets and thresholds

## API Integration

The dashboard integrates with all system components:

### Traffic Data
- **TrafficState:** Real-time traffic conditions
- **VehicleDetection:** Individual vehicle tracking
- **SignalAction:** Signal timing adjustments

### Control Systems
- **SignalControlManager:** Signal hardware interface
- **QLearningAgent:** Reinforcement learning optimization
- **PredictionEngine:** Traffic forecasting

### Data Processing
- **TrafficAggregator:** Data aggregation and analysis
- **ConfigManager:** System configuration management

## Security Features

### Access Control
- Operator ID required for manual override
- Action logging with timestamps and operator identification
- Session timeout for security

### Audit Trail
- All manual interventions logged
- Signal timing changes recorded with reasoning
- Export capabilities for compliance reporting

## Troubleshooting

### Common Issues

1. **Dashboard won't start:**
   ```bash
   # Check dependencies
   pip install -r requirements.txt
   
   # Verify configuration
   python examples/dashboard_demo.py
   ```

2. **No intersections available:**
   - Run the demo setup to create sample configuration
   - Check `config/system_config.json` for intersection definitions

3. **Real-time updates not working:**
   - Verify auto-refresh is enabled in sidebar
   - Check system component status indicators
   - Review logs for error messages

4. **Manual override not responding:**
   - Ensure operator ID is entered
   - Check for existing override sessions
   - Verify intersection selection

### Performance Optimization

- **Large datasets:** Use data filtering and pagination
- **Slow rendering:** Reduce refresh rate or chart complexity
- **Memory usage:** Clear browser cache and restart dashboard

## Development

### Adding New Components
1. Create component class in `dashboard_components.py`
2. Add rendering methods with Streamlit widgets
3. Integrate with `SystemIntegrator` for data access
4. Add to main dashboard tabs or sidebar

### Custom Visualizations
- Use Plotly for interactive charts
- Follow existing color schemes and styling
- Add responsive design for different screen sizes
- Include data export options

### Testing
```bash
# Run dashboard tests
python -m pytest tests/test_dashboard.py -v

# Run integration tests
python -m pytest tests/test_dashboard.py::TestDashboardIntegration -v
```

## Support

For technical support or questions:
- Check the main project README for system requirements
- Review logs in `logs/traffic_system.log`
- Run the demo script to verify installation
- Contact: traffic-support@city.gov

## License

This dashboard is part of the Smart Traffic Management System project. See the main project LICENSE file for details.