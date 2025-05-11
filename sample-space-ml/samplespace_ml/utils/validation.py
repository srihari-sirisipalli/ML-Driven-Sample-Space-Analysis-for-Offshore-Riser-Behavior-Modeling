# samplespace_ml/utils/validation.py
"""
Validation Utilities for checking data types, structures, and values.
"""
import pandas as pd
import numpy as np
from typing import Any, Type, Union, List, Optional, Tuple # Ensure all necessary types are imported

# Attempt to import logger; fallback to basic print/logging if not set up
try:
    from ..logging.log_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        _handler = logging.StreamHandler() # Use a temporary variable name
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)
    logger.info("samplespace_ml.logging not fully available for validation, using basic logging.")

# Assuming custom exceptions are defined
try:
    from ..exceptions.data import DataValidationError, ColumnNotFoundError, SchemaError
except ImportError:
    class DataValidationError(ValueError): pass # type: ignore
    class ColumnNotFoundError(KeyError): pass # type: ignore
    class SchemaError(ValueError): pass # type: ignore
    logger.warning("Custom data exceptions not found for validation, using basic ValueError/KeyError.")


def is_dataframe(data: Any) -> bool:
    """Checks if the provided data is a pandas DataFrame."""
    return isinstance(data, pd.DataFrame)

def is_series(data: Any) -> bool:
    """Checks if the provided data is a pandas Series."""
    return isinstance(data, pd.Series)

def is_numeric_column(series: pd.Series, col_name: Optional[str] = None) -> bool:
    """
    Checks if a pandas Series contains numeric data.

    Args:
        series (pd.Series): The Series to check.
        col_name (Optional[str]): Name of the column (for logging/error messages).

    Returns:
        bool: True if numeric, False otherwise.

    Raises:
        TypeError: If input is not a pandas Series.
    """
    if not is_series(series):
        raise TypeError(f"Input '{col_name or 'series'}' must be a pandas Series, got {type(series)}.")
    return pd.api.types.is_numeric_dtype(series)

def is_categorical_column(series: pd.Series, unique_threshold: int = 10, col_name: Optional[str] = None) -> bool:
    """
    Checks if a pandas Series is likely categorical.
    Considers dtype (object, string, category) and number of unique values for numeric types.

    Args:
        series (pd.Series): The Series to check.
        unique_threshold (int): For numeric series, if nunique <= threshold, it's considered categorical.
        col_name (Optional[str]): Name of the column (for logging/error messages).

    Returns:
        bool: True if likely categorical, False otherwise.

    Raises:
        TypeError: If input is not a pandas Series.
    """
    effective_col_name = col_name or series.name or "Unnamed Series"
    if not is_series(series):
        raise TypeError(f"Input '{effective_col_name}' must be a pandas Series, got {type(series)}.")

    if pd.api.types.is_categorical_dtype(series) or \
       pd.api.types.is_object_dtype(series) or \
       pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=False) <= unique_threshold:
        logger.debug(f"Column '{effective_col_name}' is numeric but has {series.nunique(dropna=False)} unique values (<= threshold {unique_threshold}), considered categorical.")
        return True
    return False

def check_column_exists(df: pd.DataFrame, column_name: str):
    """
    Checks if a single column exists in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        column_name (str): The name of the column to check.

    Raises:
        ColumnNotFoundError: If the column does not exist.
        TypeError: If df is not a DataFrame or column_name is not a string.
    """
    if not is_dataframe(df):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(column_name, str):
        raise TypeError("Input 'column_name' must be a string.")

    if column_name not in df.columns:
        available_cols_str = ", ".join(df.columns.astype(str).tolist())
        raise ColumnNotFoundError(f"Column '{column_name}' not found in DataFrame. Available columns: [{available_cols_str}]")

def check_columns_exist(df: pd.DataFrame, column_names: List[str]):
    """
    Checks if all columns in a list exist in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        column_names (List[str]): A list of column names.

    Raises:
        ColumnNotFoundError: If any of the specified columns do not exist.
        TypeError: If df is not a DataFrame or column_names is not a list of strings.
    """
    if not is_dataframe(df):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(column_names, list) or not all(isinstance(c, str) for c in column_names):
        raise TypeError("'column_names' must be a list of strings.")
    if not column_names: # If the list is empty, no columns to check.
        logger.debug("check_columns_exist called with an empty list of column names. No validation performed.")
        return


    missing_columns = [col for col in column_names if col not in df.columns]
    if missing_columns:
        available_cols_str = ", ".join(df.columns.astype(str).tolist())
        raise ColumnNotFoundError(
            f"Column(s) not found in DataFrame: {', '.join(missing_columns)}. Available columns: [{available_cols_str}]"
        )

def validate_input_type(value: Any, expected_type: Union[Type, Tuple[Type, ...]], arg_name: str = "argument"):
    """
    Validates if the input value is of the expected type(s).

    Args:
        value (Any): The value to validate.
        expected_type (Union[Type, Tuple[Type, ...]]): The expected type or a tuple of expected types.
        arg_name (str): The name of the argument being validated (for error messages).

    Raises:
        TypeError: If the value is not of the expected type.
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_type_names = ", ".join([t.__name__ for t in expected_type])
            msg = f"Argument '{arg_name}' must be one of types ({expected_type_names}), but got {type(value).__name__}."
        else: # Single type
            msg = f"Argument '{arg_name}' must be of type {expected_type.__name__}, but got {type(value).__name__}."
        logger.error(msg)
        raise TypeError(msg)

def validate_dataframe_not_empty(df: pd.DataFrame, df_name: str = "DataFrame"):
    """
    Validates that a DataFrame is not empty.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        df_name (str): Name of the DataFrame for error messages.

    Raises:
        DataValidationError: If the DataFrame is empty.
    """
    validate_input_type(df, pd.DataFrame, df_name)
    if df.empty:
        msg = f"{df_name} cannot be empty."
        logger.error(msg)
        raise DataValidationError(msg, validation_type="DataFrame Empty Check")

def validate_series_not_empty(series: pd.Series, series_name: str = "Series"):
    """
    Validates that a Series is not empty.

    Args:
        series (pd.Series): The Series to check.
        series_name (str): Name of the Series for error messages.

    Raises:
        DataValidationError: If the Series is empty.
    """
    validate_input_type(series, pd.Series, series_name)
    if series.empty:
        msg = f"{series_name} cannot be empty."
        logger.error(msg)
        raise DataValidationError(msg, validation_type="Series Empty Check")

def validate_numeric_range(value: Union[int, float],
                           min_val: Optional[Union[int, float]] = None,
                           max_val: Optional[Union[int, float]] = None,
                           arg_name: str = "value",
                           inclusive_min: bool = True,
                           inclusive_max: bool = True):
    """
    Validates if a numeric value falls within a specified range.

    Args:
        value (Union[int, float]): The numeric value to validate.
        min_val (Optional[Union[int, float]]): The minimum allowed value. None means no lower bound.
        max_val (Optional[Union[int, float]]): The maximum allowed value. None means no upper bound.
        arg_name (str): Name of the argument/value being checked.
        inclusive_min (bool): Whether the minimum value is inclusive.
        inclusive_max (bool): Whether the maximum value is inclusive.

    Raises:
        DataValidationError: If the value is outside the specified range.
        TypeError: If value is not numeric or bounds are not numeric if provided.
    """
    validate_input_type(value, (int, float), arg_name)
    if min_val is not None:
        validate_input_type(min_val, (int, float), f"min_val for {arg_name}")
    if max_val is not None:
        validate_input_type(max_val, (int, float), f"max_val for {arg_name}")


    error_messages = []
    if min_val is not None:
        if inclusive_min:
            if not (value >= min_val): # Using 'not >=' to handle NaNs correctly (NaN comparisons are False)
                error_messages.append(f"greater than or equal to {min_val}")
        else: # exclusive min
            if not (value > min_val):
                error_messages.append(f"strictly greater than {min_val}")
    
    if max_val is not None:
        if inclusive_max:
            if not (value <= max_val):
                error_messages.append(f"less than or equal to {max_val}")
        else: # exclusive max
            if not (value < max_val):
                error_messages.append(f"strictly less than {max_val}")

    if error_messages:
        full_error_msg = f"Argument '{arg_name}' (value: {value}) is out of bounds. Expected: " + " and ".join(error_messages) + "."
        logger.error(full_error_msg)
        raise DataValidationError(full_error_msg, validation_type="Numeric Range Check")

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    if not logger.hasHandlers():
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
        logger.setLevel(logging.DEBUG)
        logger.info("Running validation.py standalone test with basic logger.")

    # Test DataFrame
    test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C_numeric': [0.1, 0.2, 0.3]})
    test_series_numeric = pd.Series([1, 2, 2, 3, 4, 5, 100], name="NumericData")
    test_series_cat_str = pd.Series(['a', 'b', 'a', 'c'], name="StringData")
    test_series_cat_num = pd.Series([1, 2, 1, 2, 1], name="NumericAsCat") # Few unique numeric

    logger.info(f"'A' is numeric: {is_numeric_column(test_df['A'], 'A')}")
    logger.info(f"'B' is numeric: {is_numeric_column(test_df['B'], 'B')}")
    logger.info(f"'B' is categorical: {is_categorical_column(test_df['B'], col_name='B')}")
    logger.info(f"'C_numeric' is categorical (threshold 10): {is_categorical_column(test_df['C_numeric'], col_name='C_numeric')}")
    logger.info(f"'NumericAsCat' is categorical (threshold 3): {is_categorical_column(test_series_cat_num, unique_threshold=3, col_name='NumericAsCat')}")

    try:
        check_column_exists(test_df, 'A')
        logger.info("Column 'A' exists - PASSED.")
        check_columns_exist(test_df, ['A', 'B'])
        logger.info("Columns 'A', 'B' exist - PASSED.")
        # check_column_exists(test_df, 'D') # This will raise ColumnNotFoundError
    except ColumnNotFoundError as e:
        logger.error(f"Column check failed as expected: {e}")

    try:
        validate_input_type(5, str, "test_var_type")
    except TypeError as e:
        logger.error(f"Type validation failed as expected: {e}")
    validate_input_type([1,2], list, "list_var")
    logger.info("Type validation for list - PASSED")

    try:
        validate_dataframe_not_empty(pd.DataFrame(), "empty_df_test")
    except DataValidationError as e:
        logger.error(f"Empty DF validation failed as expected: {e}")

    try:
        validate_numeric_range(5, min_val=0, max_val=10, arg_name="test_num_in_range")
        logger.info("Numeric range (5 in [0,10]) - PASSED")
        validate_numeric_range(10, min_val=0, max_val=10, arg_name="test_num_at_max_inclusive")
        logger.info("Numeric range (10 in [0,10]) - PASSED")
        # validate_numeric_range(11, min_val=0, max_val=10, arg_name="test_num_over_max") # Will raise
    except DataValidationError as e:
         logger.error(f"Numeric range validation failed as expected: {e}")

    try:
        validate_numeric_range(5, min_val=5, max_val=10, inclusive_min=False, arg_name="test_num_exclusive_min_fail")
    except DataValidationError as e:
         logger.error(f"Numeric range (5 not in (5,10]) validation failed as expected: {e}")

    logger.info("validation.py standalone tests completed.")