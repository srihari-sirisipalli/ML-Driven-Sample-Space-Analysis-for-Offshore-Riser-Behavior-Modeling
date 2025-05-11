# samplespace_ml/config/config_manager.py
"""
Configuration Manager for loading and accessing YAML-based configurations.
"""
import yaml
import os
from typing import Dict, Any, Optional, Union
from collections.abc import MutableMapping # For deep merging

# Attempt to import a logger, fallback to print if not available
try:
    from ..logging.log_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers(): # Basic config if logger not set up by main app
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("samplespace_ml.logging not found or not fully initialized for ConfigManager, using basic print/logging.")


class ConfigManager:
    """
    Manages application configuration using YAML files.
    It loads a default configuration and can merge it with a user-specific configuration.
    """
    def __init__(self, default_config_path: str, user_config_path: Optional[str] = None):
        """
        Initializes the ConfigManager.

        Args:
            default_config_path (str): Path to the default YAML configuration file.
            user_config_path (Optional[str]): Path to the user-specific YAML configuration file.
                                              If None, only default config is loaded.
        """
        self.config: Dict[str, Any] = {}
        self.default_config_path: str = default_config_path
        self.user_config_path: Optional[str] = user_config_path

        if not os.path.exists(self.default_config_path):
            logger.error(f"Default configuration file not found at: {self.default_config_path}")
            # Depending on desired behavior, you might raise an error or proceed with an empty config
            # raise FileNotFoundError(f"Default configuration file not found: {self.default_config_path}")
            self.config = {} # Initialize with empty if default is missing
        else:
            self._load_default_config()

        if self.user_config_path:
            self._load_user_config() # This will merge if the file exists

        logger.info(f"ConfigManager initialized. Default: '{self.default_config_path}', User: '{self.user_config_path}'")

    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Loads a YAML file and returns its content as a dictionary."""
        logger.debug(f"Attempting to load YAML file: {file_path}")
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file not found at {file_path}. Returning empty config.")
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data is None: # Handle empty YAML file
                logger.info(f"YAML file is empty: {file_path}")
                return {}
            if not isinstance(data, dict):
                logger.error(f"YAML file content is not a dictionary: {file_path}")
                return {} # Or raise an error
            logger.info(f"Successfully loaded configuration from: {file_path}")
            return data
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}", exc_info=True)
            return {} # Or raise a custom config error
        except IOError as e:
            logger.error(f"IOError reading YAML file {file_path}: {e}", exc_info=True)
            return {} # Or raise

    def _load_default_config(self):
        """Loads the default configuration."""
        logger.debug("Loading default configuration.")
        self.config = self._load_yaml(self.default_config_path)

    def _load_user_config(self):
        """Loads user-specific configuration and merges it with defaults."""
        if self.user_config_path and os.path.exists(self.user_config_path):
            logger.debug(f"Loading user configuration from: {self.user_config_path}")
            user_config = self._load_yaml(self.user_config_path)
            if user_config: # Only merge if user_config is not empty
                self.config = self._deep_merge_configs(self.config, user_config)
                logger.info("User configuration merged with defaults.")
        elif self.user_config_path:
            logger.info(f"User configuration file not found at {self.user_config_path}, using defaults only.")


    def _deep_merge_configs(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merges the 'updates' dictionary into the 'base' dictionary.
        Modifies 'base' in place and also returns it.
        """
        for key, value in updates.items():
            if isinstance(value, MutableMapping) and key in base and isinstance(base[key], MutableMapping):
                base[key] = self._deep_merge_configs(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value for a given key path.
        Uses dot notation for nested keys (e.g., 'database.host').

        Args:
            key_path (str): The dot-separated path to the configuration key.
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The configuration value or the default.
        """
        keys = key_path.split('.')
        current_level = self.config
        try:
            for key in keys:
                if not isinstance(current_level, dict): # Path goes into a non-dict element
                    logger.debug(f"Cannot access key '{key}' in non-dictionary element for path '{key_path}'.")
                    return default
                current_level = current_level[key]
            return current_level
        except KeyError:
            logger.debug(f"Configuration key '{key_path}' not found. Returning default value: {default}")
            return default
        except TypeError: # Handles cases where current_level becomes None or non-subscriptable
            logger.debug(f"TypeError while accessing key path '{key_path}'. Returning default value: {default}")
            return default


    def set(self, key_path: str, value: Any):
        """
        Sets a configuration value for a given key path.
        Uses dot notation for nested keys. Creates nested dictionaries if they don't exist.

        Args:
            key_path (str): The dot-separated path to the configuration key.
            value (Any): The value to set.
        """
        keys = key_path.split('.')
        current_level = self.config
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})
            if not isinstance(current_level, dict):
                # This should not happen if setdefault works correctly, but as a safeguard:
                logger.error(f"Cannot create nested dictionary for key '{key}' in path '{key_path}'. Parent is not a dict.")
                return # Or raise an error
        current_level[keys[-1]] = value
        logger.info(f"Configuration key '{key_path}' set to: {value}")

    def save_user_config(self, path: Optional[str] = None, create_dir: bool = True):
        """
        Saves the current (potentially modified) configuration to the user config file.
        Only saves if a user_config_path was defined.

        Args:
            path (Optional[str]): Path to save the user config. If None, uses self.user_config_path.
            create_dir (bool): If True, creates the directory for the user config file if it doesn't exist.
        """
        save_path = path or self.user_config_path
        if not save_path:
            logger.warning("User config path not set. Cannot save user configuration.")
            # raise ValueError("User config path not set. Cannot save configuration.")
            return

        if create_dir:
            dir_name = os.path.dirname(save_path)
            if dir_name and not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name, exist_ok=True)
                    logger.info(f"Created directory for user config: {dir_name}")
                except OSError as e:
                    logger.error(f"Error creating directory {dir_name} for user config: {e}", exc_info=True)
                    return # Or raise


        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"User configuration successfully saved to: {save_path}")
        except IOError as e:
            logger.error(f"Error saving user configuration to {save_path}: {e}", exc_info=True)
            # Or raise

    def reload_config(self):
        """
        Reloads the configuration from both default and user files.
        """
        logger.info("Reloading configuration...")
        self._load_default_config()
        if self.user_config_path:
            self._load_user_config()
        logger.info("Configuration reloaded.")

    def __getitem__(self, key_path: str) -> Any:
        """Allows dictionary-like access, e.g., config_manager['database.host']"""
        value = self.get(key_path)
        if value is None: # Or some other sentinel to indicate not found
            # Mimic dict behavior for missing key
            raise KeyError(f"Configuration key '{key_path}' not found.")
        return value

    def __setitem__(self, key_path: str, value: Any):
        """Allows dictionary-like setting, e.g., config_manager['database.host'] = 'new_host'"""
        self.set(key_path, value)

    def __contains__(self, key_path: str) -> bool:
        """Allows 'in' operator, e.g., 'database.host' in config_manager"""
        return self.get(key_path) is not None # A simple check; could be more robust

    def get_all_config(self) -> Dict[str, Any]:
        """Returns a copy of the entire current configuration."""
        import copy
        return copy.deepcopy(self.config)