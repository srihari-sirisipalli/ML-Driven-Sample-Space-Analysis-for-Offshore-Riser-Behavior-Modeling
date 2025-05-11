# samplespace_ml/utils/math_tools.py
"""
Mathematical utility functions commonly used in ML and data analysis.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Any # Ensure all necessary types are imported
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # For direct use

# Assuming validation utils are in place
try:
    from .validation import validate_input_type, is_series
except ImportError: # Fallback for isolated use/testing
    def validate_input_type(value: Any, expected_type: Any, arg_name: str = "argument"):
        if not isinstance(value, expected_type): raise TypeError(f"{arg_name} type mismatch")
    def is_series(data: Any) -> bool: return isinstance(data, pd.Series)
    print("Warning: validation utils not fully imported in math_tools.py")


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
    logger.info("samplespace_ml.logging not fully available for math_tools, using basic logging.")


ArrayLike = Union[np.ndarray, pd.Series, List[Union[int, float]]]

def _ensure_numpy_arrays(*args: ArrayLike) -> Tuple[np.ndarray, ...]:
    """
    Converts inputs to flattened numpy arrays and checks for consistent length.
    If inputs are scalar, they are converted to 1-element arrays.
    """
    if not args:
        return tuple()
    
    np_arrays: List[np.ndarray] = []
    first_len: Optional[int] = None

    for i, arg in enumerate(args):
        arr: np.ndarray
        if isinstance(arg, pd.Series):
            arr = arg.to_numpy()
        elif isinstance(arg, pd.DataFrame):
            if arg.shape[1] == 1: # Single column DataFrame
                arr = arg.iloc[:, 0].to_numpy()
            else:
                raise ValueError(f"Input argument {i} (DataFrame) must have only one column to be treated as ArrayLike, got {arg.shape[1]} columns.")
        elif isinstance(arg, list):
            arr = np.array(arg)
        elif isinstance(arg, np.ndarray):
            arr = arg
        elif isinstance(arg, (int, float, np.number)): # Handle scalar inputs
             arr = np.array([arg])
        else:
            raise TypeError(f"Input argument {i} must be a list, pandas Series/DataFrame (single column), numpy array, or scalar, got {type(arg)}")
        
        # Ensure it's essentially 1D or can be treated as such
        arr = arr.squeeze() # Remove single-dimensional entries from the shape
        if arr.ndim == 0: # If squeeze results in 0-dim (scalar after squeeze)
            arr = arr.reshape(1)
        elif arr.ndim > 1:
             raise ValueError(f"Input argument {i} must be 1-dimensional or squeezable to 1D, got {arr.ndim} dimensions after squeeze.")

        if first_len is None:
            if len(arr) > 0 : # Only set first_len if array is not empty
                 first_len = len(arr)
        elif len(arr) != first_len and first_len is not None and len(arr) >0 : # compare only if both have elements
            # Special case: if one is scalar (len 1) and other is not, this is usually an error for metrics
            if first_len == 1 and len(arr) > 1 or len(arr) == 1 and first_len > 1:
                 pass # Allow broadcasting for scalar vs array if that's intended for some math ops, but metrics usually require same length
            else:
                raise ValueError(f"All input arrays/series intended for element-wise operations must have the same length. Got lengths: {first_len} and {len(arr)}.")
        elif first_len is None and len(arr) > 0: # If previous arrays were empty, set first_len
            first_len = len(arr)

        np_arrays.append(arr)
    return tuple(np_arrays)


def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        float: The RMSE value. Returns np.nan if inputs are empty or lengths mismatch after filtering NaNs.
    """
    y_true_np, y_pred_np = _ensure_numpy_arrays(y_true, y_pred)
    
    # Handle NaNs by pairwise removal
    mask = ~ (np.isnan(y_true_np) | np.isnan(y_pred_np))
    y_true_clean, y_pred_clean = y_true_np[mask], y_pred_np[mask]

    if len(y_true_clean) == 0:
        logger.warning("RMSE calculation: No valid (non-NaN) pairs of y_true and y_pred. Returning NaN.")
        return np.nan
    if len(y_true_clean) != len(y_pred_clean): # Should be caught by _ensure_numpy_arrays if initial lengths differ
        logger.error("RMSE calculation: y_true and y_pred have different lengths after NaN removal. This indicates an issue.")
        return np.nan # Or raise error

    return float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))

def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        float: The MAE value. Returns np.nan if inputs are empty or lengths mismatch after filtering NaNs.
    """
    y_true_np, y_pred_np = _ensure_numpy_arrays(y_true, y_pred)
    mask = ~ (np.isnan(y_true_np) | np.isnan(y_pred_np))
    y_true_clean, y_pred_clean = y_true_np[mask], y_pred_np[mask]

    if len(y_true_clean) == 0:
        logger.warning("MAE calculation: No valid (non-NaN) pairs of y_true and y_pred. Returning NaN.")
        return np.nan
    return float(mean_absolute_error(y_true_clean, y_pred_clean))

def calculate_r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates R-squared (Coefficient of Determination).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        float: The R-squared value. Returns np.nan if inputs are problematic.
    """
    y_true_np, y_pred_np = _ensure_numpy_arrays(y_true, y_pred)
    mask = ~ (np.isnan(y_true_np) | np.isnan(y_pred_np))
    y_true_clean, y_pred_clean = y_true_np[mask], y_pred_np[mask]

    if len(y_true_clean) < 2:
        logger.warning("R2 score is ill-defined for less than 2 valid samples. Returning NaN.")
        return np.nan
    # Check if y_true has zero variance (constant value)
    if np.var(y_true_clean) < np.finfo(float).eps: # Check for near-zero variance
         logger.warning("R2 score is not well-defined when y_true has zero variance. Returning NaN.")
         return np.nan
    return float(r2_score(y_true_clean, y_pred_clean))

def calculate_mape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles potential division by zero by various strategies.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        epsilon (float): Small constant used in handling zeros in y_true.

    Returns:
        float: The MAPE value (as a percentage). Returns np.nan if calculation is problematic.
    """
    y_true_np, y_pred_np = _ensure_numpy_arrays(y_true, y_pred)
    
    # Pairwise removal of NaNs
    nan_mask = np.isnan(y_true_np) | np.isnan(y_pred_np)
    y_true_clean = y_true_np[~nan_mask]
    y_pred_clean = y_pred_np[~nan_mask]

    if len(y_true_clean) == 0:
        logger.warning("MAPE calculation: No valid (non-NaN) pairs. Returning NaN.")
        return np.nan

    # Identify where y_true is zero or very close to zero
    zero_true_mask = np.abs(y_true_clean) < epsilon

    if np.all(zero_true_mask):
        logger.warning("MAPE calculation: All true values are zero or near zero. MAPE is undefined or infinite. Returning NaN.")
        # If all y_pred are also zero, MAPE could be 0. If y_pred are non-zero, it's infinite.
        # For simplicity and to indicate issues, returning NaN.
        return np.nan

    # Calculate absolute percentage error, avoiding division by zero
    # For elements where y_true is zero (or near zero):
    # If y_pred is also zero, error is 0.
    # If y_pred is non-zero, error is effectively infinite (or very large). We'll cap or exclude.
    
    # Option 1: Exclude true zeros from MAPE calculation (common practice)
    non_zero_y_true_mask_for_calc = np.abs(y_true_clean) > epsilon
    if not np.any(non_zero_y_true_mask_for_calc): # Should be caught by previous check, but safeguard
        return np.nan
        
    y_true_for_mape = y_true_clean[non_zero_y_true_mask_for_calc]
    y_pred_for_mape = y_pred_clean[non_zero_y_true_mask_for_calc]
    
    if len(y_true_for_mape) == 0: # All values were zero
        logger.warning("MAPE calculation: After filtering, no non-zero true values remain. Returning NaN.")
        return np.nan

    percentage_errors = np.abs((y_true_for_mape - y_pred_for_mape) / y_true_for_mape)
    mape = np.mean(percentage_errors) * 100
    
    if np.isinf(mape) or np.isnan(mape):
        logger.warning(f"MAPE calculation resulted in inf or NaN. y_true min: {y_true_clean.min()}, max: {y_true_clean.max()}. This might indicate issues with zero or very small true values.")
        return np.nan
        
    return float(mape)


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalizes a pandas Series to the range [0, 1] (Min-Max scaling).
    NaNs in the input series will result in NaNs in the output.

    Args:
        series (pd.Series): The pandas Series to normalize.

    Returns:
        pd.Series: The normalized Series.

    Raises:
        TypeError: If input is not a pandas Series.
        ValueError: If the Series is not numeric or if all non-NaN values are the same.
    """
    validate_input_type(series, pd.Series, "series")
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Series '{series.name}' must be numeric to normalize, found dtype {series.dtype}.")

    if series.isnull().all():
        logger.warning(f"Series '{series.name}' contains only NaN values. Returning a copy.")
        return series.copy()

    # Calculate min and max ignoring NaNs
    min_val = series.min(skipna=True)
    max_val = series.max(skipna=True)

    if pd.isna(min_val) or pd.isna(max_val): # Should be caught by isnull().all() if completely empty
        logger.warning(f"Series '{series.name}' has no non-NaN values to determine min/max. Returning a copy.")
        return series.copy()

    if min_val == max_val:
        # If all non-NaN values are the same, normalization results in 0 for these values.
        # Or 0.5 if one prefers to center; 0 is more standard for min-max on constant.
        logger.info(f"All non-NaN values in series '{series.name}' are identical ({min_val}). Normalized non-NaNs will be 0.0.")
        return series.apply(lambda x: 0.0 if pd.notna(x) else np.nan)

    normalized_series = (series - min_val) / (max_val - min_val)
    logger.debug(f"Series '{series.name}' normalized. Original Min: {min_val}, Max: {max_val}")
    return normalized_series

def standardize_series(series: pd.Series) -> pd.Series:
    """
    Standardizes a pandas Series (Z-score normalization: (x - mean) / std).
    NaNs in the input series will result in NaNs in the output.

    Args:
        series (pd.Series): The pandas Series to standardize.

    Returns:
        pd.Series: The standardized Series.

    Raises:
        TypeError: If input is not a pandas Series.
        ValueError: If the Series is not numeric or if its standard deviation of non-NaNs is zero.
    """
    validate_input_type(series, pd.Series, "series")
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Series '{series.name}' must be numeric to standardize, found dtype {series.dtype}.")

    if series.isnull().all():
        logger.warning(f"Series '{series.name}' contains only NaN values. Returning a copy.")
        return series.copy()

    mean_val = series.mean(skipna=True)
    std_val = series.std(skipna=True)

    if pd.isna(mean_val) or pd.isna(std_val):
        logger.warning(f"Series '{series.name}' has no non-NaN values to determine mean/std. Returning a copy.")
        return series.copy()

    if std_val == 0: # or np.isclose(std_val, 0) for float precision
        logger.info(f"Standard deviation of series '{series.name}' is zero. Standardized non-NaNs will be 0.0.")
        return series.apply(lambda x: 0.0 if pd.notna(x) else np.nan)

    standardized_series = (series - mean_val) / std_val
    logger.debug(f"Series '{series.name}' standardized. Original Mean: {mean_val}, Std: {std_val}")
    return standardized_series

def cartesian_to_polar(x: Union[float, ArrayLike],
                       y: Union[float, ArrayLike],
                       angle_units: str = 'radians') -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Converts Cartesian coordinates (x, y) to Polar coordinates (radius, angle).

    Args:
        x: X-coordinate(s). Can be scalar, list, numpy array, or pandas Series.
        y: Y-coordinate(s). Can be scalar, list, numpy array, or pandas Series.
        angle_units (str): Desired units for the angle ('radians' or 'degrees'). Default is 'radians'.

    Returns:
        tuple: (radius, angle). Type matches input (scalar or numpy array).
    """
    x_np, y_np = _ensure_numpy_arrays(x, y) # Ensures they are numpy arrays of same length

    radius = np.sqrt(x_np**2 + y_np**2)
    angle = np.arctan2(y_np, x_np) # Returns angle in radians from -pi to pi

    if angle_units.lower() == 'degrees':
        angle = np.degrees(angle)
    elif angle_units.lower() != 'radians':
        raise ValueError("angle_units must be 'radians' or 'degrees'")
    
    # If original inputs were scalars (now 1-element arrays), return scalars
    if isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)): # Check original types
        return float(radius[0]), float(angle[0])
    return radius, angle

def polar_to_cartesian(radius: Union[float, ArrayLike],
                       angle: Union[float, ArrayLike],
                       angle_units: str = 'radians') -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Converts Polar coordinates (radius, angle) to Cartesian coordinates (x, y).

    Args:
        radius: Radius value(s). Can be scalar, list, numpy array, or pandas Series.
        angle: Angle value(s). Can be scalar, list, numpy array, or pandas Series.
        angle_units (str): Units of the input angle ('radians' or 'degrees'). Default is 'radians'.

    Returns:
        tuple: (x, y). Type matches input (scalar or numpy array).
    """
    r_np, angle_np = _ensure_numpy_arrays(radius, angle) # Ensures they are numpy arrays

    if angle_units.lower() == 'degrees':
        angle_rad = np.radians(angle_np)
    elif angle_units.lower() == 'radians':
        angle_rad = angle_np
    else:
        raise ValueError("angle_units must be 'radians' or 'degrees'")

    x = r_np * np.cos(angle_rad)
    y = r_np * np.sin(angle_rad)

    if isinstance(radius, (int, float, np.number)) and isinstance(angle, (int, float, np.number)): # Check original types
        return float(x[0]), float(y[0])
    return x, y

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    if not logger.hasHandlers():
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
        logger.setLevel(logging.DEBUG)
        logger.info("Running math_tools.py standalone test with basic logger.")

    # Test metrics
    y_t = pd.Series([1, 2, 3, 4, 5, 6, np.nan, 8])
    y_p = pd.Series([1.1, 1.9, 3.3, 3.8, 5.2, 7, 7.5, np.nan])
    logger.info(f"RMSE: {calculate_rmse(y_t, y_p):.4f}")
    logger.info(f"MAE: {calculate_mae(y_t, y_p):.4f}")
    logger.info(f"R2: {calculate_r2_score(y_t, y_p):.4f}")
    logger.info(f"MAPE: {calculate_mape(y_t, y_p):.4f}%")
    
    y_t_zeros = pd.Series([0, 0, 1, 2])
    y_p_zeros = pd.Series([0.1, 0, 1.1, 2.2])
    logger.info(f"MAPE with zeros: {calculate_mape(y_t_zeros, y_p_zeros):.4f}%")


    # Test normalization and standardization
    s = pd.Series([10, 20, 30, 40, 50, np.nan, 25], name="TestSeries")
    logger.info(f"Original Series:\n{s}")
    norm_s = normalize_series(s.copy())
    logger.info(f"Normalized Series:\n{norm_s}")
    stand_s = standardize_series(s.copy())
    logger.info(f"Standardized Series:\n{stand_s}")

    s_const = pd.Series([5, 5, 5, 5, np.nan], name="ConstantSeries")
    logger.info(f"Constant Series Original:\n{s_const}")
    logger.info(f"Constant Series Normalized:\n{normalize_series(s_const.copy())}")
    logger.info(f"Constant Series Standardized:\n{standardize_series(s_const.copy())}")
    
    s_all_nan = pd.Series([np.nan, np.nan, np.nan], name="AllNaNSeries")
    logger.info(f"All NaN Series Normalized:\n{normalize_series(s_all_nan.copy())}")


    # Test coordinate transformations
    x_cart, y_cart = 1, 1
    r_pol, a_pol_rad = cartesian_to_polar(x_cart, y_cart, angle_units='radians')
    _, a_pol_deg = cartesian_to_polar(x_cart, y_cart, angle_units='degrees')
    logger.info(f"Cartesian ({x_cart},{y_cart}) -> Polar: Radius={r_pol:.4f}, AngleRad={a_pol_rad:.4f}, AngleDeg={a_pol_deg:.4f}")

    x_conv, y_conv = polar_to_cartesian(r_pol, a_pol_deg, angle_units='degrees')
    logger.info(f"Polar (R={r_pol:.4f}, A={a_pol_deg:.4f} deg) -> Cartesian: ({x_conv:.4f},{y_conv:.4f})")
    assert np.isclose(x_conv, x_cart) and np.isclose(y_conv, y_cart), "Polar to Cartesian conversion failed"

    xs = np.array([1, 0, -1, 0])
    ys = np.array([0, 1, 0, -1])
    radii, angles_deg = cartesian_to_polar(xs, ys, 'degrees')
    logger.info(f"Cartesian Arrays ({xs},{ys}) -> Polar: Radii={radii}, AnglesDeg={angles_deg}")
    x_conv_arr, y_conv_arr = polar_to_cartesian(radii, angles_deg, 'degrees')
    logger.info(f"Polar Arrays to Cartesian: ({x_conv_arr},{y_conv_arr})")
    assert np.allclose(x_conv_arr, xs) and np.allclose(y_conv_arr, ys), "Array Polar to Cartesian conversion failed"

    logger.info("math_tools.py standalone tests completed.")