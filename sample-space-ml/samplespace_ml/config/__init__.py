# samplespace_ml/config/__init__.py
"""
Configuration Module for SampleSpace ML Library
===============================================

This module handles the loading and management of application
and library configurations. It allows for default configurations
and user-overrides.

Key components:
- ConfigManager: Class to load, get, and set configuration values.
- app_config: A global instance of ConfigManager, pre-configured with
              default and user settings, to be used throughout the application.
"""

from .config_manager import ConfigManager
from .settings import app_config, DEFAULT_CONFIG_PATH, USER_CONFIG_PATH

__all__ = [
    "ConfigManager",
    "app_config",
    "DEFAULT_CONFIG_PATH",
    "USER_CONFIG_PATH"
]

print("SampleSpace ML config module initialized.")