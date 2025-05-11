# samplespace_ml/utils/__init__.py
"""
Utilities Module for SampleSpace ML Library
===========================================

This module provides common utility functions and classes used across
the SampleSpace ML library, including file I/O, validation, mathematical
tools, profiling, and threading helpers.
"""

from typing import Union, List, Dict, Any, Optional, Tuple, Callable, Type # Common typing imports

from .file_io import (
    read_csv,
    read_excel,
    save_dataframe_to_csv,
    save_dataframe_to_excel,
    save_figure,
    save_text_to_file,
    save_pickle,
    load_pickle
)
from .validation import (
    is_dataframe,
    is_series,
    is_numeric_column,
    is_categorical_column,
    check_column_exists,
    check_columns_exist,
    validate_input_type,
    validate_dataframe_not_empty,
    validate_series_not_empty,
    validate_numeric_range # Added
)
from .math_tools import (
    calculate_rmse,
    calculate_mae,
    calculate_r2_score,
    calculate_mape,
    normalize_series,
    standardize_series,
    cartesian_to_polar,
    polar_to_cartesian
)
from .profiler import profile_function, PerformanceProfiler
from .threading import BaseWorkerThread, PYQT5_AVAILABLE # Expose PYQT5_AVAILABLE

__all__ = [
    # file_io
    "read_csv",
    "read_excel",
    "save_dataframe_to_csv",
    "save_dataframe_to_excel",
    "save_figure",
    "save_text_to_file",
    "save_pickle",
    "load_pickle",
    # validation
    "is_dataframe",
    "is_series",
    "is_numeric_column",
    "is_categorical_column",
    "check_column_exists",
    "check_columns_exist",
    "validate_input_type",
    "validate_dataframe_not_empty",
    "validate_series_not_empty",
    "validate_numeric_range",
    # math_tools
    "calculate_rmse",
    "calculate_mae",
    "calculate_r2_score",
    "calculate_mape",
    "normalize_series",
    "standardize_series",
    "cartesian_to_polar",
    "polar_to_cartesian",
    # profiler
    "profile_function",
    "PerformanceProfiler",
    # threading
    "BaseWorkerThread",
    "PYQT5_AVAILABLE" # Useful for conditional GUI features
]

print("SampleSpace ML utils module initialized.")