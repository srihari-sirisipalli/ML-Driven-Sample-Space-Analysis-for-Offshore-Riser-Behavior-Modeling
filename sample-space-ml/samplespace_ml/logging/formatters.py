# samplespace_ml/logging/formatters.py
"""
Custom logging formatters for the SampleSpace ML library.
"""
import logging
import json # For JsonFormatter
from typing import Dict, Any, Optional
import sys

class SimpleFormatter(logging.Formatter):
    """
    A simple log formatter, typically showing level and message.
    Example: INFO: This is an info message.
    """
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, validate: bool = True):
        custom_fmt = fmt or "%(levelname)s: %(message)s"
        super().__init__(fmt=custom_fmt, datefmt=datefmt, validate=validate)


class DetailedFormatter(logging.Formatter):
    """
    A more detailed log formatter.
    Includes timestamp, logger name, level, module, function, line number, and message.
    """
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, validate: bool = True):
        default_fmt = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
        default_datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt=fmt or default_fmt, datefmt=datefmt or default_datefmt, validate=validate)


class JsonFormatter(logging.Formatter):
    """
    Formats log records as JSON strings.
    Useful for structured logging, especially when sending logs to services
    like Elasticsearch, Splunk, etc.
    """
    def __init__(self,
                 fmt_dict: Optional[Dict[str, str]] = None,
                 datefmt: Optional[str] = None,
                 extra_attrs: Optional[Dict[str, Any]] = None):
        """
        Args:
            fmt_dict (Optional[Dict[str, str]]): A dictionary mapping desired JSON keys
                to LogRecord attributes (e.g., {"level": "levelname", "msg": "message"}).
                If None, uses a default set of LogRecord attributes.
            datefmt (Optional[str]): Date format string for the timestamp.
            extra_attrs (Optional[Dict[str, Any]]): Static key-value pairs to add to every log record.
        """
        super().__init__()
        self.datefmt = datefmt or "%Y-%m-%d %H:%M:%S" # Changed to a valid strftime format
        self.default_fmt_dict = {
            "timestamp": "asctime",
            "level": "levelname",
            "logger_name": "name",
            "module": "module",
            "function": "funcName",
            "line": "lineno",
            "message": "message",
            "thread_id": "thread",
            "thread_name": "threadName",
            "process_id": "process",
        }
        self.fmt_dict = fmt_dict if fmt_dict is not None else self.default_fmt_dict
        self.extra_attrs = extra_attrs if extra_attrs is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the LogRecord instance into a JSON string.
        """
        log_entry: Dict[str, Any] = self.extra_attrs.copy()

        for json_key, record_attr_name in self.fmt_dict.items():
            val = getattr(record, record_attr_name, None)
            if record_attr_name == "asctime": # Special handling for asctime
                val = self.formatTime(record, self.datefmt)
            elif record_attr_name == "message":
                val = record.getMessage() # Ensures args are interpolated
            log_entry[json_key] = val

        # Handle exception information
        if record.exc_info:
            log_entry["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            log_entry["exception_message"] = str(record.exc_info[1]) if record.exc_info[1] else None
            log_entry["exception_traceback"] = self.formatException(record.exc_info)

        # Handle stack information
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        # Include any extra fields passed to the logger
        # These are fields not part of standard LogRecord attributes
        standard_attrs = set(record.__dict__.keys())
        custom_extras = {k: v for k, v in record.__dict__.items() if k not in standard_attrs and k not in ['args', 'msg', 'exc_text']} # filter some internals
        if custom_extras:
            log_entry["extra"] = custom_extras
            
        # Ensure all values are serializable
        try:
            return json.dumps(log_entry, default=str) # Use str as default for non-serializable
        except TypeError:
            # Fallback for very problematic objects, log the issue and a simplified record
            problematic_keys = []
            for key, value in log_entry.items():
                try:
                    json.dumps({key: value}, default=str)
                except TypeError:
                    problematic_keys.append(key)
                    log_entry[key] = f"<UnserializableObject type='{type(value).__name__}'>"
            
            # Log a warning about unserializable content (using a basic formatter to avoid recursion)
            basic_formatter = logging.Formatter()
            record.msg = (f"JsonFormatter: Could not serialize fields: {problematic_keys}. "
                          f"Original message: {record.getMessage()}")
            return json.dumps(log_entry, default=str)


# Example usage (more practical in log_config.py when setting up handlers)
if __name__ == '__main__':
    # Setup a basic logger for testing formatters
    logger = logging.getLogger("formatter_test_app")
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)

    print("\n--- Testing SimpleFormatter ---")
    console_handler.setFormatter(SimpleFormatter())
    logger.info("This is an info message.")
    logger.warning("This is a warning.")

    print("\n--- Testing DetailedFormatter ---")
    console_handler.setFormatter(DetailedFormatter())
    logger.debug("This is a debug message with details.")
    logger.error("This is an error with details.")

    print("\n--- Testing JsonFormatter ---")
    console_handler.setFormatter(JsonFormatter(extra_attrs={"app_version": "1.0-test"}))
    logger.info("User login attempt.", extra={"username": "test_user", "ip_address": "192.168.1.100"})
    try:
        value = 10 / 0
    except ZeroDivisionError:
        logger.exception("A division error occurred during JSON formatting test.")
    
    # Remove handler to prevent interference if other tests run
    logger.removeHandler(console_handler)