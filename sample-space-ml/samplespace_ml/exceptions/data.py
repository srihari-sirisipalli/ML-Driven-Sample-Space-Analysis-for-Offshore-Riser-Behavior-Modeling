# samplespace_ml/exceptions/data.py
"""
Custom exceptions related to data processing, loading, validation, and I/O.
"""

if __name__ == '__main__':
    # Use relative import with correct path when running directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from samplespace_ml.exceptions.base import SampleSpaceMLError
else:
    # Use relative import when imported as a module
    from .base import SampleSpaceMLError

from typing import Optional, List, Any, Dict, Union

class DataError(SampleSpaceMLError):
    """Base class for data-related errors within the SampleSpace ML library."""
    def __init__(self, message: str, file_path: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception, context)
        if file_path:
            self.add_context("file_path", file_path)

class FileOperationError(DataError):
    """Raised for errors during file reading or writing operations."""
    def __init__(self, message: str, file_path: str, operation: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, file_path, original_exception, context)
        if operation:
            self.add_context("operation", operation) # e.g., 'read', 'write', 'open'

class UnsupportedFileTypeError(DataError):
    """Raised when an unsupported file type is encountered for an operation."""
    def __init__(self, file_type: str, supported_types: Optional[List[str]] = None, file_path: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        message = f"Unsupported file type: '{file_type}'."
        if supported_types:
            message += f" Supported types are: {', '.join(supported_types)}."
        super().__init__(message, file_path, original_exception, context)
        self.add_context("unsupported_type", file_type)
        if supported_types:
            self.add_context("supported_types", supported_types)


class DataValidationError(DataError):
    """Raised when data validation fails (e.g., wrong format, integrity issues)."""
    def __init__(self, message: str, validation_type: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception=original_exception, context=context)
        if validation_type:
            self.add_context("validation_type", validation_type)

class SchemaError(DataValidationError):
    """Raised for issues related to data schema (e.g., missing columns, wrong dtypes)."""
    def __init__(self, message: str, schema_details: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, validation_type="SchemaValidation", original_exception=original_exception, context=context)
        if schema_details:
            self.add_context("schema_details", schema_details)

class ColumnNotFoundError(SchemaError):
    """Raised when an expected column or columns are not found in a DataFrame."""
    def __init__(self, column_names: Union[str, List[str]], message: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        names_str = column_names if isinstance(column_names, str) else ", ".join(column_names)
        custom_message = message or f"Column(s) not found: '{names_str}'."
        super().__init__(custom_message, original_exception=original_exception, context=context)
        self.add_context("missing_columns", column_names)

class PreprocessingError(DataError):
    """Raised for errors during data preprocessing steps like cleaning or transformation."""
    def __init__(self, message: str, step_name: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception=original_exception, context=context)
        if step_name:
            self.add_context("preprocessing_step", step_name)


if __name__ == '__main__':
    # Example usage and testing of data-related exceptions
    
    # Test FileOperationError
    try:
        raise FileNotFoundError("test.csv not found")
    except FileNotFoundError as e:
        file_error = FileOperationError(
            message="Failed to read data file",
            file_path="test.csv",
            operation="read",
            original_exception=e
        )
        print("FileOperationError example:")
        print(file_error)
        print()

    # Test UnsupportedFileTypeError
    supported_formats = ['.csv', '.parquet', '.json']
    unsupported_error = UnsupportedFileTypeError(
        file_type='.xlsx',
        supported_types=supported_formats,
        file_path="data.xlsx"
    )
    print("UnsupportedFileTypeError example:")
    print(unsupported_error)
    print()

    # Test ColumnNotFoundError
    missing_cols = ['timestamp', 'pressure']
    col_error = ColumnNotFoundError(
        column_names=missing_cols,
        context={'available_columns': ['temperature', 'depth']}
    )
    print("ColumnNotFoundError example:")
    print(col_error)
    print()

    # Test PreprocessingError
    try:
        raise ValueError("Invalid value encountered during normalization")
    except ValueError as e:
        prep_error = PreprocessingError(
            message="Data normalization failed",
            step_name="normalize_features",
            original_exception=e,
            context={'affected_columns': ['pressure', 'temperature']}
        )
        print("PreprocessingError example:")
        print(prep_error)