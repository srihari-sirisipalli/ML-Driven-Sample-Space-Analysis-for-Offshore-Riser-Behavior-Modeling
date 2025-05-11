# samplespace_ml/utils/file_io.py
"""
File I/O Utilities for reading and saving various file types.
"""
import pandas as pd
import os
import pickle
from typing import Optional, Dict, Any, Union, List # Ensure Union and List are imported
import matplotlib.pyplot as plt # For type hinting plt.Figure

# Attempt to import logger; fallback to basic print/logging if not set up
try:
    from ..logging.log_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__) # Get a default logger
    if not logger.hasHandlers(): # Add a basic handler if none are configured
        _handler = logging.StreamHandler() # Use a temporary variable name
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)
    logger.info("samplespace_ml.logging not fully available for file_io, using basic logging.")

# Assuming custom exceptions are defined
try:
    from ..exceptions.data import FileOperationError, UnsupportedFileTypeError
except ImportError:
    # Basic fallback exceptions if custom ones are not available
    class FileOperationError(IOError): # type: ignore
        def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None, original_exception: Optional[Exception] = None):
            super().__init__(message)
            self.file_path = file_path
            self.operation = operation
            self.original_exception = original_exception

    class UnsupportedFileTypeError(ValueError): # type: ignore
        def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None, original_exception: Optional[Exception] = None):
            super().__init__(message)
            self.file_path = file_path
            self.operation = operation
            self.original_exception = original_exception
    logger.warning("Custom data exceptions not found for file_io, using basic IOError/ValueError.")


def read_csv(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileOperationError: If the file is not found or cannot be read.
    """
    if not os.path.exists(file_path):
        msg = f"CSV file not found: {file_path}"
        logger.error(msg)
        raise FileOperationError(msg, file_path=file_path, operation="read_csv")
    try:
        logger.info(f"Reading CSV file: {file_path}")
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        msg = f"Error reading CSV file {file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise FileOperationError(msg, file_path=file_path, operation="read_csv", original_exception=e)

def read_excel(file_path: str, sheet_name: Optional[Union[str, int, List[Union[str, int]]]] = 0, **kwargs: Any) -> Union[pd.DataFrame, Dict[Union[str, int], pd.DataFrame]]:
    """
    Reads an Excel file into a pandas DataFrame or a dictionary of DataFrames.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (Optional[Union[str, int, List[Union[str, int]]]]):
            Name(s) or index/indices of the sheet(s) to read.
            Defaults to 0 (the first sheet). If None, reads all sheets.
        **kwargs: Additional keyword arguments to pass to pandas.read_excel.

    Returns:
        Union[pd.DataFrame, Dict[Union[str, int], pd.DataFrame]]:
            A DataFrame if a single sheet is read, or a dictionary of DataFrames
            if multiple or all sheets are read.

    Raises:
        FileOperationError: If the file is not found or cannot be read.
    """
    if not os.path.exists(file_path):
        msg = f"Excel file not found: {file_path}"
        logger.error(msg)
        raise FileOperationError(msg, file_path=file_path, operation="read_excel")
    try:
        logger.info(f"Reading Excel file: {file_path}, sheet(s): {sheet_name}")
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    except Exception as e:
        msg = f"Error reading Excel file {file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise FileOperationError(msg, file_path=file_path, operation="read_excel", original_exception=e)

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool = False, **kwargs: Any):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to save the CSV file.
        index (bool): Whether to write DataFrame index as a column (default is False).
        **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_csv.

    Raises:
        FileOperationError: If the DataFrame cannot be saved.
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    try:
        logger.info(f"Saving DataFrame (shape: {df.shape}) to CSV: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
        df.to_csv(file_path, index=index, **kwargs)
        logger.info(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        msg = f"Error saving DataFrame to CSV {file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise FileOperationError(msg, file_path=file_path, operation="save_csv", original_exception=e)

def save_dataframe_to_excel(df: pd.DataFrame, file_path: str, sheet_name: str = 'Sheet1', index: bool = False, **kwargs: Any):
    """
    Saves a pandas DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to save the Excel file.
        sheet_name (str): Name of the sheet to write to (default is 'Sheet1').
        index (bool): Whether to write DataFrame index as a column (default is False).
        **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_excel.

    Raises:
        FileOperationError: If the DataFrame cannot be saved.
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    try:
        logger.info(f"Saving DataFrame (shape: {df.shape}) to Excel: {file_path}, sheet: {sheet_name}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
        df.to_excel(file_path, sheet_name=sheet_name, index=index, **kwargs)
        logger.info(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        msg = f"Error saving DataFrame to Excel {file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise FileOperationError(msg, file_path=file_path, operation="save_excel", original_exception=e)

def save_figure(fig: plt.Figure, file_path: str, dpi: int = 300, **kwargs: Any):
    """
    Saves a Matplotlib figure to a file.

    Args:
        fig (plt.Figure): The Matplotlib figure object to save.
        file_path (str): The path to save the figure.
        dpi (int): Dots per inch for the saved figure (default is 300).
        **kwargs: Additional keyword arguments to pass to fig.savefig.

    Raises:
        FileOperationError: If the figure cannot be saved.
        UnsupportedFileTypeError: If the file extension is not recognized or supported by Matplotlib.
        TypeError: If fig is not a Matplotlib Figure.
    """
    if not isinstance(fig, plt.Figure): # Check specific Figure type from matplotlib
        raise TypeError("Input 'fig' must be a Matplotlib Figure instance.")

    file_dir = os.path.dirname(file_path)
    if file_dir: # Ensure directory exists only if path includes a directory
        os.makedirs(file_dir, exist_ok=True)

    _, ext = os.path.splitext(file_path)
    if not ext: # Add a default extension if none provided
        logger.warning(f"No extension provided for saving figure. Defaulting to '.png'. Path will be: {file_path}.png")
        file_path += ".png"

    try:
        logger.info(f"Saving figure to: {file_path} with DPI: {dpi}")
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight', **kwargs)
        logger.info(f"Figure successfully saved to {file_path}")
    except Exception as e:
        msg = f"Error saving figure to {file_path}: {e}"
        logger.error(msg, exc_info=True)
        # Check for common Matplotlib error for unsupported types
        if "unknown file extension" in str(e).lower() or \
           "no handler for file extension" in str(e).lower():
            raise UnsupportedFileTypeError(msg, file_path=file_path, operation="save_figure", original_exception=e)
        raise FileOperationError(msg, file_path=file_path, operation="save_figure", original_exception=e)

def save_text_to_file(text_content: str, file_path: str, encoding: str = 'utf-8'):
    """
    Saves text content to a file.

    Args:
        text_content (str): The string content to save.
        file_path (str): The path to save the text file.
        encoding (str): The file encoding to use (default is 'utf-8').

    Raises:
        FileOperationError: If the text cannot be saved.
        TypeError: If text_content is not a string.
    """
    if not isinstance(text_content, str):
        raise TypeError("Input 'text_content' must be a string.")
    try:
        logger.info(f"Saving text content to: {file_path} with encoding: {encoding}")
        file_dir = os.path.dirname(file_path)
        if file_dir: # Ensure directory exists
            os.makedirs(file_dir, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(text_content)
        logger.info(f"Text successfully saved to {file_path}")
    except Exception as e:
        msg = f"Error saving text to {file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise FileOperationError(msg, file_path=file_path, operation="save_text", original_exception=e)

def save_pickle(obj: Any, file_path: str, protocol: int = pickle.HIGHEST_PROTOCOL, **kwargs: Any):
    """
    Saves a Python object to a pickle file.

    Args:
        obj (Any): The Python object to save.
        file_path (str): The path to save the pickle file.
        protocol (int): Pickle protocol to use. Defaults to highest available.
        **kwargs: Additional keyword arguments for pickle.dump().

    Raises:
        FileOperationError: If the object cannot be pickled.
    """
    try:
        logger.info(f"Saving object of type {type(obj).__name__} to pickle file: {file_path} using protocol {protocol}")
        file_dir = os.path.dirname(file_path)
        if file_dir: # Ensure directory exists
            os.makedirs(file_dir, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=protocol, **kwargs)
        logger.info(f"Object successfully pickled to {file_path}")
    except Exception as e:
        msg = f"Error saving object to pickle file {file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise FileOperationError(msg, file_path=file_path, operation="save_pickle", original_exception=e)

def load_pickle(file_path: str, **kwargs: Any) -> Any:
    """
    Loads a Python object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.
        **kwargs: Additional keyword arguments for pickle.load().

    Returns:
        Any: The loaded Python object.

    Raises:
        FileOperationError: If the file is not found or the object cannot be unpickled.
    """
    if not os.path.exists(file_path):
        msg = f"Pickle file not found: {file_path}"
        logger.error(msg)
        raise FileOperationError(msg, file_path=file_path, operation="load_pickle")
    try:
        logger.info(f"Loading object from pickle file: {file_path}")
        with open(file_path, 'rb') as f:
            obj = pickle.load(f, **kwargs)
        logger.info(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        msg = f"Error loading object from pickle file {file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise FileOperationError(msg, file_path=file_path, operation="load_pickle", original_exception=e)

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    # Create a dummy logger for standalone testing if main logging isn't set up
    if not logger.hasHandlers():
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)
        logger.info("Running file_io.py standalone test with basic logger.")

    # Create a test directory
    test_dir = "temp_file_io_test"
    os.makedirs(test_dir, exist_ok=True)
    logger.info(f"Created test directory: {test_dir}")

    # Test DataFrame
    data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
    test_df = pd.DataFrame(data)

    # Test CSV
    csv_path = os.path.join(test_dir, "test.csv")
    try:
        save_dataframe_to_csv(test_df, csv_path, index=True)
        loaded_csv_df = read_csv(csv_path, index_col=0)
        assert test_df.equals(loaded_csv_df), "CSV read/write failed"
        logger.info(f"CSV read/write test successful: {csv_path}")
    except Exception as e:
        logger.error(f"CSV test failed: {e}")

    # Test Excel
    excel_path = os.path.join(test_dir, "test.xlsx")
    try:
        save_dataframe_to_excel(test_df, excel_path, sheet_name="TestSheet")
        loaded_excel_df = read_excel(excel_path, sheet_name="TestSheet")
        assert test_df.equals(loaded_excel_df), "Excel read/write failed"
        logger.info(f"Excel read/write test successful: {excel_path}")
    except Exception as e:
        logger.error(f"Excel test failed: {e}")

    # Test Text
    text_path = os.path.join(test_dir, "test.txt")
    try:
        save_text_to_file("Hello SampleSpace ML!", text_path)
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == "Hello SampleSpace ML!", "Text read/write failed"
        logger.info(f"Text read/write test successful: {text_path}")
    except Exception as e:
        logger.error(f"Text test failed: {e}")

    # Test Pickle
    pickle_path = os.path.join(test_dir, "test.pkl")
    test_obj = {"key": "value", "numbers": [1, 2, 3]}
    try:
        save_pickle(test_obj, pickle_path)
        loaded_obj = load_pickle(pickle_path)
        assert test_obj == loaded_obj, "Pickle read/write failed"
        logger.info(f"Pickle read/write test successful: {pickle_path}")
    except Exception as e:
        logger.error(f"Pickle test failed: {e}")

    # Test Figure (basic)
    fig_path = os.path.join(test_dir, "test_figure.png")
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")
        save_figure(fig, fig_path)
        assert os.path.exists(fig_path), "Figure save failed"
        logger.info(f"Figure save test successful: {fig_path}")
        plt.close(fig) # Close the figure
    except Exception as e:
        logger.error(f"Figure save test failed: {e}")

    logger.info("file_io.py standalone tests completed.")
    # print(f"\nConsider manually deleting the test directory: {test_dir}") # Optional cleanup