"""Data models for the Smart Traffic Management System."""

from .traffic_state import TrafficState
from .vehicle_detection import VehicleDetection
from .signal_action import SignalAction

__all__ = ['TrafficState', 'VehicleDetection', 'SignalAction']