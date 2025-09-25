"""Unit tests for utility functions."""

import pytest
import logging
import tempfile
import os
from src.utils import setup_logging, TrafficSystemError, ValidationError, ConfigurationError
from src.utils.error_handling import handle_error, safe_execute


class TestLoggingConfig:
    """Test cases for logging configuration."""
    
    def test_setup_logging_console_only(self):
        """Test setting up logging with console output only."""
        setup_logging(log_level="DEBUG")
        
        logger = logging.getLogger("test_logger")
        assert logger.getEffectiveLevel() == logging.DEBUG
    
    def test_setup_logging_with_file(self):
        """Test setting up logging with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            setup_logging(log_level="INFO", log_file=log_file)
            
            logger = logging.getLogger("test_logger")
            logger.info("Test message")
            
            # Close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)
            
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_invalid_log_level(self):
        """Test setup with invalid log level defaults to INFO."""
        setup_logging(log_level="INVALID")
        
        logger = logging.getLogger("test_logger")
        assert logger.getEffectiveLevel() == logging.INFO


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_traffic_system_error(self):
        """Test TrafficSystemError creation and properties."""
        error = TrafficSystemError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert str(error) == "Test error"
    
    def test_validation_error(self):
        """Test ValidationError creation and properties."""
        error = ValidationError(
            message="Invalid field value",
            field_name="test_field",
            invalid_value="invalid"
        )
        
        assert error.message == "Invalid field value"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.field_name == "test_field"
        assert error.invalid_value == "invalid"
    
    def test_configuration_error(self):
        """Test ConfigurationError creation and properties."""
        error = ConfigurationError(
            message="Missing configuration",
            config_section="database"
        )
        
        assert error.message == "Missing configuration"
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.config_section == "database"
    
    def test_handle_error_with_traffic_system_error(self, caplog):
        """Test error handling with TrafficSystemError."""
        error = TrafficSystemError("Test error", "TEST_CODE")
        
        with caplog.at_level(logging.ERROR):
            handle_error(error, "test context")
        
        assert "Traffic system error in test context: Test error" in caplog.text
    
    def test_handle_error_with_generic_exception(self, caplog):
        """Test error handling with generic exception."""
        error = ValueError("Generic error")
        
        with caplog.at_level(logging.ERROR):
            handle_error(error, "test context")
        
        assert "Unexpected error in test context: Generic error" in caplog.text
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        def test_function(x, y):
            return x + y
        
        result = safe_execute(test_function, 2, 3, context="test")
        assert result == 5
    
    def test_safe_execute_with_exception(self, caplog):
        """Test safe_execute with function that raises exception."""
        def failing_function():
            raise ValueError("Function failed")
        
        with caplog.at_level(logging.ERROR):
            result = safe_execute(
                failing_function,
                default_return="default",
                context="test context"
            )
        
        assert result == "default"
        assert "Unexpected error in test context" in caplog.text
    
    def test_safe_execute_with_kwargs(self):
        """Test safe_execute with keyword arguments."""
        def test_function(x, y=10):
            return x * y
        
        result = safe_execute(test_function, 3, context="test", y=4)
        assert result == 12