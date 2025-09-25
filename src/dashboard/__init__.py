"""Smart Traffic Management System Dashboard Module."""

try:
    from .simple_dashboard import SimpleTrafficDashboard
    __all__ = ['SimpleTrafficDashboard']
except ImportError:
    # Fallback if imports fail
    __all__ = []