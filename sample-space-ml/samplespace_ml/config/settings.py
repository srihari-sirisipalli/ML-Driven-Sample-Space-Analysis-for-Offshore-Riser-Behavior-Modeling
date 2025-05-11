# samplespace_ml/config/settings.py
"""
Global application settings and configuration access.

This module initializes and provides a global instance of the ConfigManager,
making configuration settings easily accessible throughout the application.
"""
import os
from typing import Optional, Dict
from .config_manager import ConfigManager

# --- Determine Paths ---
# Path to the root of the 'samplespace_ml' package
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the 'config' directory within the package
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Default configuration file, expected to be alongside this settings.py
DEFAULT_CONFIG_FILENAME = 'default_config.yml'
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, DEFAULT_CONFIG_FILENAME)

# User-specific configuration (optional)
# Stored in the user's home directory in a dedicated folder for the application.
# This allows users to override default settings without modifying package files.
USER_CONFIG_DIR_NAME = ".samplespace_ml"  # Hidden directory in user's home
USER_CONFIG_FILENAME = "user_config.yml"
USER_HOME_DIR = os.path.expanduser("~")
USER_CONFIG_FULL_DIR = os.path.join(USER_HOME_DIR, USER_CONFIG_DIR_NAME)

# Ensure the user config directory exists if we plan to write to it
# This is usually done when saving, but can be done at init for read attempts too.
# if not os.path.exists(USER_CONFIG_FULL_DIR):
#     try:
#         os.makedirs(USER_CONFIG_FULL_DIR, exist_ok=True)
#     except OSError:
#         # Handle potential error during directory creation, e.g., permissions
#         print(f"Warning: Could not create user config directory at {USER_CONFIG_FULL_DIR}")
#         # Proceed without user config or log more severely

USER_CONFIG_PATH = os.path.join(USER_CONFIG_FULL_DIR, USER_CONFIG_FILENAME)


# --- Initialize Global ConfigManager ---
# This `app_config` instance will be imported by other modules to get settings.
try:
    app_config = ConfigManager(
        default_config_path=DEFAULT_CONFIG_PATH,
        user_config_path=USER_CONFIG_PATH
    )
except FileNotFoundError as e:
    # This might happen if default_config.yml is missing during ConfigManager init
    print(f"Critical Error: Default configuration file is missing. {e}")
    print("Please ensure 'default_config.yml' exists in the config directory.")
    # Fallback to an empty config manager or raise the error to halt execution
    app_config = ConfigManager(default_config_path="dummy_non_existent_path.yml") # Will be empty
    # raise # Or re-raise to stop the application

# --- Convenience Functions to Access Common Settings ---
# These can provide type checking or more specific default handling if needed.

def get_logging_settings() -> dict:
    """Returns logging configuration dictionary."""
    return app_config.get('logging', {})

def get_logging_level(default_level: str = 'INFO') -> str:
    """Returns the configured logging level."""
    return app_config.get('logging.level', default_level).upper()

def get_data_loader_settings() -> dict:
    """Returns data loader specific settings."""
    return app_config.get('data_loader', {})

def get_visualization_theme(default_theme: str = 'default_theme') -> str:
    """Returns the default theme for visualizations."""
    return app_config.get('visualization.default_theme', default_theme)

def get_model_hyperparameters(model_name: str, default_params: Optional[dict] = None) -> dict:
    """
    Retrieves default hyperparameters for a given model.
    Example path in YAML: models.<model_name>.hyperparameters
    """
    if default_params is None:
        default_params = {}
    return app_config.get(f'models.{model_name}.hyperparameters', default_params)

def get_api_keys() -> dict:
    """Example: Retrieves API keys if stored in config."""
    return app_config.get('api_keys', {})


# --- Environment Variable Overrides (Example) ---
# You might want to allow environment variables to override config file settings.
# Example:
# DEBUG_MODE_ENV = os.environ.get('SAMPLESPACE_DEBUG_MODE')
# if DEBUG_MODE_ENV is not None:
#     app_config.set('debug_mode', DEBUG_MODE_ENV.lower() == 'true')

# --- Application Constants (derived from config or hardcoded if truly constant) ---
APP_NAME = app_config.get('application.name', "SampleSpace ML Tool")
APP_VERSION = app_config.get('application.version', "0.1.0") # Or import from __init__.py

# Example of how another module would use this:
# from samplespace_ml.config.settings import app_config, get_logging_level
# current_log_level = get_logging_level()
# some_value = app_config.get('some_section.some_key', 'default_value')