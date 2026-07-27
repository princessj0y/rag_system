import logging
from tqdm import tqdm
import contextvars
from contextlib import contextmanager, asynccontextmanager

# Module-level context storage (thread and async-safe)
mdc_context = contextvars.ContextVar("mdc_context", default={})

# Filter to integrate TQDM, see https://github.com/tqdm/tqdm/blob/master/tqdm/contrib/logging.py
class TqdmStreamHandler(logging.Handler):
    """
    A secure logging handler that pushes records straight into tqdm.write() 
    so logs don't shred progress bars, completely preserving filters.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

# Filter to inject context into log records
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

class MDCFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "mdc"):
            record.mdc = ""
        return super().format(record)

# Setup Logging
handler = TqdmStreamHandler()
handler.setFormatter(MDCFormatter('%(asctime)s | %(levelname)s%(mdc)s | %(message)s'))
handler.addFilter(MDCFilter())

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(handler)
root_logger.setLevel(logging.WARN)

logger = logging.getLogger("my_app")
logger.setLevel(logging.INFO)
# Ensure messages bubble up to the root handler we configured
logger.propagate = True

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

@asynccontextmanager
async def async_mdc(**kwargs):
    """
    Async context manager to temporarily set MDC values.
    Usage: async with mdc(request_id="123"): ...
    """
    # Capture the current state to restore it later
    token = mdc_context.set({**mdc_context.get(), **kwargs})
    
    try:
        yield
    finally:
        # Reset the context back to exactly what it was before the 'async with' block
        mdc_context.reset(token)