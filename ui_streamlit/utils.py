import time
import streamlit as st
import functools

def notify_timing(message_prefix):
    """
    Decorator to measure execution time and show a Streamlit toast notification.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                st.toast(f"✅ {message_prefix} in {duration:.2f} s")
                return result
            except Exception as e:
                # Optionally notify of error too? 
                # Better to let the original error handling handle it.
                raise e
        return wrapper
    return decorator
