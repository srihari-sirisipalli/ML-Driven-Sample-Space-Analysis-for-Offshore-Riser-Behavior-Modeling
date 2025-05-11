# samplespace_ml/exceptions/__init__.py
"""
Custom Exceptions for SampleSpace ML Library
============================================

This module defines custom exception classes to provide more specific
error information throughout the library. This helps in debugging
and allows for more granular error handling by the application
or other library components.

Exposed Exceptions:
- SampleSpaceMLError: The base exception for this library.
- DataError: Base for data-related issues.
  - FileOperationError: For file read/write problems.
  - UnsupportedFileTypeError: For unsupported file formats.
  - DataValidationError: For general data integrity issues.
  - SchemaError: For mismatches in expected data structure.
  - ColumnNotFoundError: When a required column is missing.
  - PreprocessingError: For errors during data cleaning or transformation.
- ModelError: Base for model-related issues.
  - ModelTrainingError: For errors during model fitting.
  - ModelPredictionError: For errors during prediction.
  - ModelNotFittedError: When predict/transform is called on an unfitted model.
  - IncompatibleModelError: For mismatches between model and data/task.
  - HyperparameterError: For invalid model hyperparameters.
  - ModelSerializationError: For issues saving or loading models.
- UIError: Base for GUI-related issues (relevant if GUI tools use these).
  - WidgetConfigurationError: For problems with UI widget setup.
  - UserActionError: When a user action in the UI leads to an error.
  - PlottingError: For errors during UI plot generation.
"""

from typing import Union, List, Dict, Any, Optional, Tuple, Callable, Type

from .base import SampleSpaceMLError
from .data import (
    DataError,
    FileOperationError,
    UnsupportedFileTypeError,
    DataValidationError,
    SchemaError,
    ColumnNotFoundError,
    PreprocessingError
)
from .model import (
    ModelError,
    ModelTrainingError,
    ModelPredictionError,
    ModelNotFittedError,
    IncompatibleModelError,
    HyperparameterError,
    ModelSerializationError
)
from .ui import (
    UIError,
    WidgetConfigurationError,
    UserActionError,
    PlottingError
)

__all__ = [
    "SampleSpaceMLError",
    "DataError",
    "FileOperationError",
    "UnsupportedFileTypeError",
    "DataValidationError",
    "SchemaError",
    "ColumnNotFoundError",
    "PreprocessingError",
    "ModelError",
    "ModelTrainingError",
    "ModelPredictionError",
    "ModelNotFittedError",
    "IncompatibleModelError",
    "HyperparameterError",
    "ModelSerializationError",
    "UIError",
    "WidgetConfigurationError",
    "UserActionError",
    "PlottingError"
]

# This print statement is for confirming module initialization during development.
print("SampleSpace ML exceptions module initialized.")