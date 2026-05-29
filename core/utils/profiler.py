import time
import functools
import os
import sys
import logging

# Set up logging to both a file and standard output
log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cortex_performance.log")

# Configure logger
logger = logging.getLogger("cortex_profiler")
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if already configured
if not logger.handlers:
    # File handler
    try:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Profiler could not initialize file logging: {e}")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_environment():
    # Detect Streamlit Community Cloud environment
    is_streamlit_cloud = (
        os.getenv("STREAMLIT_SERVER_SHARING_MEDIA_PATH") is not None or
        os.getenv("STREAMLIT_RUNTIME_ENV") is not None or
        "STREAMLIT_SHARING" in os.environ or
        os.path.expanduser("~").startswith("/home/appuser")
    )
    if is_streamlit_cloud:
        return "STREAMLIT_CLOUD"
    return "LOCAL"

def profile_func(func):
    """
    Decorator to log execution time of functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        env = get_environment()
        start_time = time.perf_counter()
        
        # Log arguments briefly for context (omitting path names or very long inputs)
        arg_summary = []
        if args:
            # First arg might be path, take basename if string
            first_arg = args[0]
            if isinstance(first_arg, str) and ("\\" in first_arg or "/" in first_arg):
                arg_summary.append(os.path.basename(first_arg))
            else:
                arg_summary.append(str(first_arg)[:30])
        if kwargs:
            for k, v in list(kwargs.items())[:2]:
                arg_summary.append(f"{k}={str(v)[:20]}")
        
        args_str = ", ".join(arg_summary)
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(f"[{env}] SUCCESS | {func.__module__}.{func.__name__}({args_str}) took {duration:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(f"[{env}] FAILURE | {func.__module__}.{func.__name__}({args_str}) failed after {duration:.4f} seconds with error: {e}")
            raise e
            
    return wrapper
