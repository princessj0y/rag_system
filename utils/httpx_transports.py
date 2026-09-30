import httpx
import asyncio
from .httpx_logging import default_sync_httpx_transport, default_async_httpx_transport

class SyncKeyRotationHttpxTransport(httpx.BaseTransport):

    def __init__(self, shuffled_keys, delegate: httpx.BaseTransport = None):
        super().__init__()
        self.keys = shuffled_keys
        self.total_keys = len(shuffled_keys)
        self.current_index = 0
        self.delegate = delegate or default_sync_httpx_transport(delegate)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        while True:
            active_key = self.keys[self.current_index]
            request.headers["Authorization"] = f"Bearer {active_key}"
            
            try:
                response = self.delegate.handle_request(request)
            except Exception as e:
                raise e
            
            if response.status_code != 429:
                return response
            
            self.current_index += 1
            if self.current_index >= self.total_keys:
                self.current_index = 0
                raise httpx.HTTPStatusError("Ollama keys completely exhausted.", request=request, response=response)
    
    def close(self) -> None:
        self.delegate.close()

class AsyncKeyRotationHttpxTransport(httpx.AsyncBaseTransport):

    def __init__(self, shuffled_keys, delegate: httpx.AsyncBaseTransport = None):
        super().__init__()
        self.keys = shuffled_keys
        self.total_keys = len(shuffled_keys)
        self.current_index = 0
        self.delegate = delegate or default_async_httpx_transport(delegate)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        while True:
            active_key = self.keys[self.current_index]
            request.headers["Authorization"] = f"Bearer {active_key}"
            try:
                response = await self.delegate.handle_async_request(request)
            except Exception as e:
                raise e
            
            if response.status_code != 429:
                return response
            
            self.current_index += 1
            if self.current_index >= self.total_keys:
                self.current_index = 0
                raise httpx.HTTPStatusError("Ollama keys completely exhausted.", request=request, response=response)

    async def aclose(self) -> None:
        """Ensure the underlying delegate transport is gracefully closed."""
        await self.delegate.aclose()

class BoundedAsyncHttpxTransport(httpx.AsyncBaseTransport):
    """
    A decorator wrapper for any httpx Async Transport that chokes concurrency
    using a shared asyncio.Semaphore before delegating the HTTP call.
    """
    def __init__(self, semaphore: asyncio.Semaphore, delegate: httpx.AsyncBaseTransport = None):
        super().__init__()
        self.delegate = delegate or default_async_httpx_transport(delegate)
        self.semaphore = semaphore

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self.semaphore:
            # Delegate the actual HTTP call to the underlying transport instance
            return await self.delegate.handle_async_request(request)

    async def aclose(self) -> None:
        """Ensure the underlying delegate transport is gracefully closed."""
        await self.delegate.aclose()
