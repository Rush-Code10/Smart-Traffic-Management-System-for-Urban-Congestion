"""Error handling and custom exceptions for the traffic management system."""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class TrafficSystemError(Exception):
    """Base exception class for traffic management system errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)
        
        # Log the error
        logger.error(f"TrafficSystemError: {message} (Code: {error_code})")


class ValidationError(TrafficSystemError):
    """Exception raised when data validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, invalid_value: Optional[Any] = None):
        self.field_name = field_name
        self.invalid_value = invalid_value
        
        error_details = {
            'field_name': field_name,
            'invalid_value': invalid_value
        }
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class ConfigurationError(TrafficSystemError):
    """Exception raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_section: Optional[str] = None):
        self.config_section = config_section
        
        error_details = {
            'config_section': config_section
        }
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=error_details
        )


class CameraProcessingError(TrafficSystemError):
    """Exception raised when camera processing fails."""
    
    def __init__(self, message: str, camera_id: Optional[str] = None):
        self.camera_id = camera_id
        
        error_details = {
            'camera_id': camera_id
        }
        
        super().__init__(
            message=message,
            error_code="CAMERA_PROCESSING_ERROR",
            details=error_details
        )


class SignalControlError(TrafficSystemError):
    """Exception raised when signal control operations fail."""
    
    def __init__(self, message: str, intersection_id: Optional[str] = None):
        self.intersection_id = intersection_id
        
        error_details = {
            'intersection_id': intersection_id
        }
        
        super().__init__(
            message=message,
            error_code="SIGNAL_CONTROL_ERROR",
            details=error_details
        )


def handle_error(error: Exception, context: str = "") -> None:
    """
    Handle and log errors with appropriate context.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    """
    if isinstance(error, TrafficSystemError):
        logger.error(f"Traffic system error in {context}: {error.message}")
        if error.details:
            logger.error(f"Error details: {error.details}")
    else:
        logger.error(f"Unexpected error in {context}: {str(error)}", exc_info=True)


def safe_execute(func, *args, default_return=None, context: str = "", **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if function fails
        context: Context description for error logging
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result or default_return if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, context)
        return default_return