import uuid
import logging
import contextvars
from contextlib import contextmanager

# 1. Module-level context storage (thread and async-safe)
mdc_context = contextvars.ContextVar("mdc_context", default={})

# 2. Filter to inject context into log records
class MDCFilter(logging.Filter):
    def filter(self, record):
        context = mdc_context.get()
        
        if context:
            # Format the active context items into a single string
            # e.g., "[request_id: 123] [user: alice]"
            parts = [f"[{k}: {v}]" for k, v in context.items()]
            record.mdc = " | " + " | ".join(parts)
        else:
            # If no context, leave it as an empty string
            record.mdc = ""
        
        return True

# 3. Setup Logging
logger = logging.getLogger("my_app")
handler = logging.StreamHandler()

# Ensure placeholders match your context keys
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s%(mdc)s | %(message)s'
)
handler.setFormatter(formatter)
handler.addFilter(MDCFilter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Helper to update context
def set_mdc(**kwargs):
    new_context = mdc_context.get().copy()
    new_context.update(kwargs)
    mdc_context.set(new_context)

@contextmanager
def mdc(**kwargs):
    """
    Context manager to temporarily set MDC values.
    Usage: with mdc(request_id="123"): ...
    """
    # 1. Capture the current state to restore it later
    token = mdc_context.set({**mdc_context.get(), **kwargs})
    
    try:
        yield
    finally:
        # 2. Reset the context back to exactly what it was before the 'with' block
        mdc_context.reset(token)