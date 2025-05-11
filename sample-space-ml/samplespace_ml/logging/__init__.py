# samplespace_ml/logging/__init__.py
"""
Logging Module for SampleSpace ML Library
=========================================

Handles application-wide logging configuration, providing functions
to set up logging handlers and retrieve logger instances. It also
includes custom formatters for log messages.

Key components:
- setup_logging: Configures root logger with console and optional file handlers.
- get_logger: Retrieves a named logger instance.
- SimpleFormatter, DetailedFormatter, JsonFormatter: Custom log formatters.
"""

from typing import Union, List, Dict, Any, Optional, Tuple, Callable, Type

from .formatters import SimpleFormatter, DetailedFormatter, JsonFormatter
from .log_config import setup_logging, get_logger


__all__ = [
    "setup_logging",
    "get_logger",
    "SimpleFormatter",
    "DetailedFormatter",
    "JsonFormatter"
]

# This print statement is for confirming module initialization during development.
# It might be removed or logged via the logger itself in a production setting
# once the logging system is bootstrapped.
print("SampleSpace ML logging module initialized.")