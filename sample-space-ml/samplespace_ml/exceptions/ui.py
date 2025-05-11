# samplespace_ml/exceptions/ui.py
"""
Custom exceptions related to the user interface (GUI) components
of tools built with SampleSpace ML.
"""

if __name__ == '__main__':
    # Use absolute import when running directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from samplespace_ml.exceptions.base import SampleSpaceMLError
else:
    # Use relative import when imported as a module
    from .base import SampleSpaceMLError

from typing import Optional, Any, Dict

class UIError(SampleSpaceMLError):
    """Base class for UI-related errors within SampleSpace ML tools."""
    def __init__(self, message: str, widget_name: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception, context)
        if widget_name:
            self.add_context("widget_name", widget_name)

class WidgetConfigurationError(UIError):
    """Raised when a UI widget is improperly configured or required data for it is missing."""
    def __init__(self, message: str, widget_name: Optional[str] = None, configuration_detail: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, widget_name, original_exception, context)
        if configuration_detail:
            self.add_context("configuration_detail", configuration_detail)

class UserActionError(UIError):
    """
    Raised when a user action (e.g., button click) cannot be completed due to
    an invalid application state or user input.
    """
    def __init__(self, message: str, action_name: Optional[str] = None, reason: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception=original_exception, context=context)
        if action_name:
            self.add_context("action_name", action_name)
        if reason:
            self.add_context("reason", reason)

class PlottingError(UIError):
    """Raised for errors specifically occurring during plot generation for the UI."""
    def __init__(self, message: str, plot_type: Optional[str] = None, original_exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_exception=original_exception, context=context)
        if plot_type:
            self.add_context("plot_type", plot_type)


if __name__ == '__main__':
    # Example usage of UI-related exceptions
    try:
        # Simulate a widget configuration error
        raise WidgetConfigurationError(
            message="Required data source not configured",
            widget_name="DataPlotWidget",
            configuration_detail="data_source parameter is missing"
        )
    except WidgetConfigurationError as e:
        print(f"Caught WidgetConfigurationError: {str(e)}")
        print(f"Context: {e.context}")

    try:
        # Simulate a user action error
        raise UserActionError(
            message="Cannot execute requested action",
            action_name="export_plot",
            reason="No plot data available to export"
        )
    except UserActionError as e:
        print(f"Caught UserActionError: {str(e)}")
        print(f"Context: {e.context}")

    try:
        # Simulate a plotting error
        raise PlottingError(
            message="Failed to generate scatter plot",
            plot_type="scatter",
            original_exception=ValueError("x and y arrays must have the same length")
        )
    except PlottingError as e:
        print(f"Caught PlottingError: {str(e)}")
        print(f"Context: {e.context}")