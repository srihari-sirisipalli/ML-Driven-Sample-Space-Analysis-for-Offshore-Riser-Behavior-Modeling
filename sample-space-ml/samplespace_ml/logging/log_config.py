# samplespace_ml/logging/log_config.py
"""
Logging configuration for the application.
Provides functions to set up and retrieve logger instances.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, Any

# Attempt to import the main app_config
try:
    from ..config.settings import app_config
except ImportError:
    # Fallback DummyConfig if the main config system isn't available
    # This is primarily for isolated testing or if the module is used standalone.
    class DummyConfigManagerForLogging:
        def get(self, key: str, default: Any = None) -> Any:
            config_map = {
                'logging.level': 'INFO',
                'logging.format': "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
                'logging.date_format': "%Y-%m-%d %H:%M:%S",
                'logging.log_file': None,
                'logging.console_output': True,
                'logging.log_to_file_enabled': False
            }
            return config_map.get(key, default)
    app_config = DummyConfigManagerForLogging() # type: ignore
    # Use basic print for this warning as logger might not be set up
    print("Warning: samplespace_ml.config.settings.app_config not found. Using fallback logging defaults for logging setup.")

# Attempt to import custom formatters, fallback to a basic one
try:
    from .formatters import DetailedFormatter, SimpleFormatter, JsonFormatter
except ImportError:
    print("Warning: Custom formatters from .formatters not found. Defining a basic DetailedFormatter locally.")
    class DetailedFormatter(logging.Formatter): # type: ignore
        """A basic detailed formatter for log messages."""
        def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: str = '%', validate: bool = True):
            default_fmt = "%(asctime)s - %(name)s - %(levelname)-8s - [%(filename)s:%(lineno)d] (%(funcName)s) - %(message)s"
            default_datefmt = "%Y-%m-%d %H:%M:%S"
            try: # Python 3.8+
                super().__init__(fmt=fmt or default_fmt, datefmt=datefmt or default_datefmt, style=style, validate=validate)
            except TypeError: # Older Python versions
                 super().__init__(fmt=fmt or default_fmt, datefmt=datefmt or default_datefmt)
    # Define other formatters as None or basic if needed
    SimpleFormatter = DetailedFormatter # Fallback
    JsonFormatter = DetailedFormatter   # Fallback


# Define constants for default logging parameters
DEFAULT_LOG_LEVEL_NAME: str = "INFO"
# This will be overridden by app_config if available and `log_format_override` is None
DEFAULT_LOG_FORMAT_FALLBACK: str = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
DEFAULT_DATE_FORMAT_FALLBACK: str = "%Y-%m-%d %H:%M:%S"
ROOT_LOGGER_NAME: str = "samplespace_ml"

_logging_configured_by_samplespace = False

def setup_logging(
    log_level_override: Optional[Union[int, str]] = None,
    log_format_override: Optional[str] = None,
    date_format_override: Optional[str] = None,
    log_file_override: Optional[str] = None,
    console_output_override: Optional[bool] = None,
    log_to_file_override: Optional[bool] = None,
    formatter_instance: Optional[logging.Formatter] = None, # Allow passing a formatter instance
    force_reconfigure: bool = False
):
    """
    Sets up logging for the SampleSpace ML library.
    Reads configuration from `app_config` by default.
    """
    global _logging_configured_by_samplespace

    # Get settings from config, allowing overrides
    level_name = log_level_override or app_config.get('logging.level', DEFAULT_LOG_LEVEL_NAME)
    log_fmt = log_format_override or app_config.get('logging.format', DEFAULT_LOG_FORMAT_FALLBACK)
    date_fmt = date_format_override or app_config.get('logging.date_format', DEFAULT_DATE_FORMAT_FALLBACK)
    log_file_path = log_file_override or app_config.get('logging.log_file')
    console_enabled = console_output_override if console_output_override is not None \
        else app_config.get('logging.console_output', True)
    file_logging_enabled = log_to_file_override if log_to_file_override is not None \
        else app_config.get('logging.log_to_file_enabled', False)

    numeric_level = getattr(logging, str(level_name).upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Warning: Invalid log level '{level_name}'. Defaulting to INFO.")
        numeric_level = logging.INFO

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False

    if force_reconfigure or not _logging_configured_by_samplespace or not logger.hasHandlers():
        if logger.hasHandlers(): # Clear only if forcing or if it's the first *real* setup
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        if force_reconfigure:
             print(f"Info: Forced reconfiguration of logging for '{ROOT_LOGGER_NAME}'.")

        current_formatter = formatter_instance or DetailedFormatter(fmt=log_fmt, datefmt=date_fmt)
        handlers_added = False

        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(current_formatter)
            logger.addHandler(console_handler)
            handlers_added = True

        if file_logging_enabled and log_file_path:
            try:
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    log_file_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
                )
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(current_formatter)
                logger.addHandler(file_handler)
                handlers_added = True
                if not _logging_configured_by_samplespace or force_reconfigure:
                    logger.info(f"File logging to '{log_file_path}' enabled at level {level_name}.")
            except Exception as e:
                print(f"ERROR: Failed to set up file logging to '{log_file_path}': {e}")
                logger.error(f"Failed to set up file logging to '{log_file_path}': {e}", exc_info=True)
                if not console_enabled:
                    console_handler_fb = logging.StreamHandler(sys.stdout)
                    console_handler_fb.setLevel(numeric_level)
                    console_handler_fb.setFormatter(current_formatter)
                    logger.addHandler(console_handler_fb)
                    logger.warning("File logging failed. Outputting to console as a fallback.")
                    handlers_added = True
        elif file_logging_enabled and not log_file_path:
            logger.warning("File logging is enabled in config, but 'logging.log_file' path is not set.")

        if not handlers_added:
            logger.addHandler(logging.NullHandler())
            if not _logging_configured_by_samplespace or force_reconfigure:
                 logger.debug(f"Logging for '{ROOT_LOGGER_NAME}' initialized with NullHandler.")
        else:
            if not _logging_configured_by_samplespace or force_reconfigure:
                logger.info(f"Logging for '{ROOT_LOGGER_NAME}' (re)initialized. Level: {level_name}.")
        
        _logging_configured_by_samplespace = True
    else:
        logger.debug(f"Logging for '{ROOT_LOGGER_NAME}' already configured. Current level: {logging.getLevelName(logger.level)}.")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger instance.
    """
    global _logging_configured_by_samplespace
    if not _logging_configured_by_samplespace:
        # Basic setup if not called explicitly, ensures loggers don't break.
        # Uses defaults from app_config (which might be DummyConfig).
        print(f"Info: Auto-initializing logging for '{ROOT_LOGGER_NAME}' on first get_logger call.")
        setup_logging()

    if name:
        return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(ROOT_LOGGER_NAME)


if __name__ == "__main__":
    print("Running log_config.py self-test...")
    # Define a temporary app_config for testing purposes
    class TestConfigManager:
        def get(self, key: str, default: Any = None) -> Any:
            test_config_map = {
                'logging.level': 'DEBUG',
                'logging.format': "%(asctime)s [%(levelname)-5s] %(name)s (%(funcName)s): %(message)s",
                'logging.date_format': "%H:%M:%S",
                'logging.log_file': "temp_samplespace_test.log",
                'logging.console_output': True,
                'logging.log_to_file_enabled': True
            }
            return test_config_map.get(key, default)
    app_config = TestConfigManager() # type: ignore # Override global app_config for this test block

    print("\n--- Test 1: Initial setup with test config ---")
    setup_logging(force_reconfigure=True)
    
    main_logger = get_logger()
    main_logger.info("Main logger info after initial setup.")

    test_module_logger = get_logger("test_module.submodule")
    test_module_logger.debug("This is a debug message for test_module.submodule.")
    test_module_logger.info("This is an info message for test_module.submodule.")
    test_module_logger.warning("This is a warning for test_module.submodule.")

    print("\n--- Test 2: Get logger (should use existing setup) ---")
    another_logger = get_logger("another_module")
    another_logger.info("Info message from another_logger, should use existing setup.")

    print("\n--- Test 3: Error logging ---")
    try:
        x = 1 / 0
    except ZeroDivisionError:
        test_module_logger.error("A division by zero error occurred.", exc_info=True)
        test_module_logger.critical("This is a critical error after the exception.")

    print(f"\nLog messages should be in console and '{app_config.get('logging.log_file')}'") # type: ignore
    print("Self-test finished.")

    # Clean up test log file
    if os.path.exists("temp_samplespace_test.log"):
        # Ensure handlers are closed before trying to delete
        for handler in main_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        # os.remove("temp_samplespace_test.log") # Commented out to allow inspection
        print("Test log file 'temp_samplespace_test.log' created. Please inspect/delete manually.")