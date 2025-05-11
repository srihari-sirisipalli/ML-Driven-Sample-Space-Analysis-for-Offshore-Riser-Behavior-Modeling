# samplespace_ml/exceptions/model.py
"""
Custom exceptions related to machine learning models, their training,
prediction, and management.
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

from typing import Optional, Any, Dict

class ModelError(SampleSpaceMLError):
    """Base class for model-related errors within the SampleSpace ML library."""
    def __init__(self, message: str, model_name: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception, context)
        if model_name:
            self.add_context("model_name", model_name)

class ModelTrainingError(ModelError):
    """Raised when an error occurs during the model training (fitting) process."""
    def __init__(self, message: str, model_name: Optional[str] = None, training_stage: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_name, original_exception, context)
        if training_stage:
            self.add_context("training_stage", training_stage)

class ModelPredictionError(ModelError):
    """Raised when an error occurs during the model prediction phase."""
    def __init__(self, message: str, model_name: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_name, original_exception, context)

class ModelNotFittedError(ModelError):
    """Raised when attempting to use a model that has not been fitted yet."""
    def __init__(self, model_name: str = "Model", operation: str = "predict/transform"):
        message = f"{model_name} has not been fitted. Call fit() before attempting to {operation}."
        super().__init__(message, model_name)
        self.add_context("operation_attempted", operation)

class IncompatibleModelError(ModelError):
    """
    Raised when a model is used in an incompatible way, such as with
    incorrect input data features or for an unsuitable task.
    """
    def __init__(self, message: str, model_name: Optional[str] = None, reason: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_name, original_exception, context)
        if reason:
            self.add_context("incompatibility_reason", reason)

class HyperparameterError(ModelError):
    """Raised for invalid hyperparameter configurations or values."""
    def __init__(self, message: str, model_name: Optional[str] = None, parameter_name: Optional[str] = None, invalid_value: Any = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_name, original_exception, context)
        if parameter_name:
            self.add_context("parameter_name", parameter_name)
        if invalid_value is not None: # Check for None explicitly
            self.add_context("invalid_value", str(invalid_value)) # Convert to string for context

class ModelSerializationError(ModelError):
    """Raised for errors occurring during model saving (serialization) or loading (deserialization)."""
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception=original_exception, context=context)
        if file_path:
            self.add_context("file_path", file_path)
        if operation:
            self.add_context("serialization_operation", operation) # e.g., 'save', 'load'

if __name__ == '__main__':
    # Example usage of the custom exceptions
    try:
        # Simulate a model not fitted error
        raise ModelNotFittedError("RandomForestRegressor", "predict")
    except ModelNotFittedError as e:
        print(f"Caught ModelNotFittedError: {str(e)}")
        print(f"Context: {e.context}")

    try:
        # Simulate a hyperparameter error
        raise HyperparameterError(
            message="Invalid learning rate value",
            model_name="GradientBoostingRegressor",
            parameter_name="learning_rate",
            invalid_value=-0.1
        )
    except HyperparameterError as e:
        print(f"Caught HyperparameterError: {str(e)}")
        print(f"Context: {e.context}")