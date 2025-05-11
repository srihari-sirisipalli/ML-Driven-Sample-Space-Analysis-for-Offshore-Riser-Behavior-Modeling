# samplespace_ml/utils/profiler.py
"""
Profiling utilities for measuring code execution time and performance.
"""
import time
import cProfile
import pstats
import io
from functools import wraps
from typing import Callable, Any, Optional, List, Union, Tuple, Type # Ensure Type is imported

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
    logger.info("samplespace_ml.logging not fully available for profiler, using basic logging.")


SUPPORTED_PSTATS_SORT_KEYS: List[str] = [
    'calls',      # call count
    'cumulative', # cumulative time
    'cumtime',    # cumulative time
    'file',       # file name
    'filename',   # file name
    'line',       # line number
    'module',     # file name
    'name',       # function name
    'ncalls',     # call count
    'pcalls',     # primitive call count
    'stdname',    # standard name
    'time',       # internal time
    'tottime'     # internal time
]

def profile_function(sort_by: Union[str, List[str]] = 'cumulative',
                     restrictions: Union[List[Any], Tuple[Any, ...], int, float, str] = (),
                     output_to_console: bool = True):
    """
    A decorator to profile a function's execution using cProfile.

    Args:
        sort_by (Union[str, List[str]]): Stat(s) to sort profiling output by.
            Valid options are from `pstats.Stats.sort_stats()`.
            Defaults to 'cumulative'.
        restrictions (Union[List[Any], Tuple[Any, ...], int, float, str]):
            cProfile restrictions (e.g., module names, line numbers, regex for function names,
            or an integer for number of lines to print, or a float for percentage of lines).
            See pstats.print_stats() documentation for details.
            Example: `profile_function(restrictions=[MyClass.__module__])`
        output_to_console (bool): If True, prints profiling results to console.
                                 Results are always logged via the logger.
    """
    # Validate sort_by keys
    sort_by_validated: Union[str, Tuple[str, ...]]
    if isinstance(sort_by, str):
        if sort_by not in SUPPORTED_PSTATS_SORT_KEYS:
            logger.warning(f"Invalid sort_by key '{sort_by}' for cProfile. Defaulting to 'cumulative'. Valid keys: {SUPPORTED_PSTATS_SORT_KEYS}")
            sort_by_validated = 'cumulative'
        else:
            sort_by_validated = sort_by
    elif isinstance(sort_by, list):
        valid_keys_in_list = [s for s in sort_by if s in SUPPORTED_PSTATS_SORT_KEYS]
        if len(valid_keys_in_list) != len(sort_by):
            logger.warning(f"Some invalid sort_by keys in list {sort_by} for cProfile. Using valid ones: {valid_keys_in_list} or defaulting if none valid.")
        sort_by_validated = tuple(valid_keys_in_list) if valid_keys_in_list else 'cumulative' # pstats takes tuple for multi-sort
    else: # Should not happen with type hints, but as a safeguard
        logger.warning(f"Unsupported type for sort_by: {type(sort_by)}. Defaulting to 'cumulative'.")
        sort_by_validated = 'cumulative'


    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = cProfile.Profile()
            logger.info(f"Starting cProfile for function: {func.__name__}")
            profiler.enable()
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by_validated)
                
                # Convert single restriction to tuple if needed for *restrictions
                actual_restrictions = restrictions
                if not isinstance(restrictions, (list, tuple)):
                    actual_restrictions = (restrictions,) # type: ignore

                ps.print_stats(*actual_restrictions) # type: ignore

                log_output = f"\n--- cProfile results for {func.__name__} (sorted by: {sort_by_validated}, restrictions: {restrictions}) ---\n"
                log_output += s.getvalue()
                log_output += f"--- End of cProfile for {func.__name__} ---\n"
                
                logger.info(log_output)
                if output_to_console:
                    print(log_output)
            return result
        return wrapper
    return decorator

class PerformanceProfiler:
    """
    A context manager and simple timer for profiling blocks of code using time.perf_counter().
    Provides more granular timing than cProfile for specific code sections.
    Supports nesting and logs indentation for readability.
    """
    _active_profilers_stack: List[str] = [] # Class variable to track nested profilers for indentation

    def __init__(self, name: str = "Unnamed Block", verbose: bool = True, log_level: int = logging.INFO):
        """
        Args:
            name (str): A descriptive name for the code block being profiled.
            verbose (bool): If True, logs start and end messages with timings.
            log_level (int): Logging level for the profiling messages (e.g., logging.DEBUG, logging.INFO).
        """
        self.name = name
        self.verbose = verbose
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
        self._indent_str: str = ""

    def __enter__(self):
        self._indent_str = "  " * len(PerformanceProfiler._active_profilers_stack)
        if self.verbose:
            logger.log(self.log_level, f"{self._indent_str}Starting: '{self.name}'...")
        PerformanceProfiler._active_profilers_stack.append(self.name)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[Any]): # Standard type for traceback object
        self.end_time = time.perf_counter()
        
        # Pop from stack after calculating indent for the exit message
        if PerformanceProfiler._active_profilers_stack and \
           PerformanceProfiler._active_profilers_stack[-1] == self.name:
            PerformanceProfiler._active_profilers_stack.pop()
        else:
            # This indicates a mismatched enter/exit, which shouldn't happen with `with` statement
            logger.warning(f"Profiler stack mismatch on exit for '{self.name}'. Stack: {PerformanceProfiler._active_profilers_stack}")


        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
            if self.verbose:
                status_msg = "completed" if exc_type is None else f"failed ({exc_type.__name__})"
                logger.log(self.log_level,
                           f"{self._indent_str}Finished: '{self.name}' [{status_msg}] in {self.elapsed_time:.6f} seconds")
        else:
            if self.verbose:
                logger.warning(f"{self._indent_str}Could not measure performance for '{self.name}', start_time was not set.")
        
        # Return False (or None) to not suppress the exception if one occurred
        return False

    def get_elapsed_time(self) -> Optional[float]:
        """Returns the elapsed time in seconds if the block has finished execution."""
        return self.elapsed_time

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    # Ensure basic logging is configured if running this file standalone
    if not logger.hasHandlers():
        _h = logging.StreamHandler() # temp var
        _f = logging.Formatter('%(asctime)s - %(levelname)-7s - %(name)s - %(message)s') # temp var
        _h.setFormatter(_f)
        logger.addHandler(_h)
        logger.propagate = False # Don't send to root if we add handler here
        logger.setLevel(logging.DEBUG) # Set to DEBUG to see profiler debug messages
        logger.info("Running profiler.py standalone test with basic DEBUG logger.")

    @profile_function(sort_by='tottime', restrictions=10) # Sort by total time, show top 10 lines
    def example_heavy_computation(n_loops: int):
        logger.info(f"Running example_heavy_computation with {n_loops} loops")
        total = 0
        with PerformanceProfiler("Main Computation Loop", log_level=logging.DEBUG) as main_loop_timer:
            for i in range(n_loops):
                with PerformanceProfiler(f"Inner Calculation {i}", verbose=False) as inner_timer: # Less verbose inner
                    # Simulate some work
                    temp_list = [j*j for j in range(1000)]
                    total += sum(temp_list)
                    if i % (n_loops // 5 or 1) == 0 : # Log progress occasionally
                         logger.debug(f"Loop {i}: Current total: {total}, Inner time: {inner_timer.get_elapsed_time():.6f}s")
            
            # Simulate another step
            with PerformanceProfiler("Final Aggregation", log_level=logging.DEBUG):
                time.sleep(0.01)
                final_result = total / (n_loops or 1)
                logger.debug(f"Final aggregation result: {final_result}")

        logger.info(f"Total time for Main Computation Loop: {main_loop_timer.get_elapsed_time():.4f}s")
        return final_result

    logger.info("Starting profiler examples.")
    result = example_heavy_computation(20) # Reduced loops for quicker test
    logger.info(f"Result of profiled function: {result}")

    # Test standalone PerformanceProfiler
    with PerformanceProfiler("Quick Standalone Task") as timer:
        time.sleep(0.005)
        x = [1] * 100000
        del x
    if timer.get_elapsed_time() is not None:
        logger.info(f"Quick Standalone Task elapsed time: {timer.get_elapsed_time():.6f} seconds")
    
    logger.info("profiler.py standalone tests completed.")