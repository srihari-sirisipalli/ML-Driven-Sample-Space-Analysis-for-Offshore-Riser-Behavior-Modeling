# samplespace_ml/exceptions/base.py
"""
Base custom exception for the SampleSpace ML library.
All other custom exceptions in this library should inherit from SampleSpaceMLError.
"""
from typing import Optional, Any, Dict

class SampleSpaceMLError(Exception):
    """
    Base class for all custom exceptions in the SampleSpace ML library.

    Attributes:
        message (str): The error message.
        original_exception (Optional[Exception]): The original exception that was caught, if any.
        context (Optional[Dict[str, Any]]): Additional context about the error.
    """
    def __init__(self, message: str, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """
        Args:
            message (str): A human-readable error message.
            original_exception (Optional[Exception]): The underlying exception, if this error
                                                     is being raised in response to another.
            context (Optional[Dict[str, Any]]): A dictionary for additional contextual
                                                information about the error.
        """
        super().__init__(message)
        self.message: str = message
        self.original_exception: Optional[Exception] = original_exception
        self.context: Dict[str, Any] = context if context is not None else {}

    def __str__(self) -> str:
        """
        Returns a string representation of the exception.
        Includes the original exception message if present.
        """
        full_message = self.message
        if self.original_exception:
            full_message += f"\n  Caused by: {type(self.original_exception).__name__}: {str(self.original_exception)}"
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            full_message += f"\n  Context: {{{context_str}}}"
        return full_message

    def add_context(self, key: str, value: Any):
        """Adds a key-value pair to the error context."""
        self.context[key] = value
        return self # Allow chaining

if __name__ == '__main__':
    # Example usage and testing
    try:
        # Simulate an error condition
        raise ValueError("Something went wrong")
    except ValueError as e:
        # Create a SampleSpaceMLError with context
        error = SampleSpaceMLError(
            message="Failed to process data",
            original_exception=e,
            context={"operation": "data_processing", "status": "failed"}
        )
        # Add additional context
        error.add_context("timestamp", "2023-12-25")
        
        # Print the error to test string representation
        print(error)