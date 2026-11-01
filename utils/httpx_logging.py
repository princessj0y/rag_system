import os
import json
from datetime import datetime
import httpx
import traceback
from utils.my_log import mdc_context

is_http_logging_enabled = os.getenv("DEBUG_HTTP_LOGS", "false").lower() == "true"
HTTP_LOG_DIR = None

if is_http_logging_enabled:
    startup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    HTTP_LOG_DIR = os.path.join(os.getcwd(), f"logs/http_logs_{startup_ts}")
    os.makedirs(HTTP_LOG_DIR, exist_ok=True)

def default_async_httpx_transport(delegate: httpx.AsyncBaseTransport = None) -> httpx.AsyncBaseTransport:
    """Returns the base transport, wrapped with logging if enabled."""
    base = delegate or httpx.AsyncHTTPTransport()
    if is_http_logging_enabled:
        return FileLoggingAsyncTransport(delegate=base)
    return base

def default_sync_httpx_transport(delegate: httpx.BaseTransport = None) -> httpx.BaseTransport:
    """Returns the base transport, wrapped with logging if enabled."""
    base = delegate or httpx.HTTPTransport()
    if is_http_logging_enabled:
        return FileLoggingSyncTransport(delegate=base)
    return base

###############################################################################
#                              LOGGING                                        #
###############################################################################


def _write_log_file(
        request: httpx.Request, 
        req_body: bytes, 
        response: httpx.Response = None, 
        res_body: bytes = b"", 
        exception: Exception = None,
):
    """Writes the formatted request and response pair to a new timestamped file."""
    if not HTTP_LOG_DIR: 
        return

    # Clearly mark errors (exceptions or non-2xx status codes)
    is_error = exception is not None or (response and not (200 <= response.status_code < 300))
    
    # Microsecond precision ensures parallel requests get unique files
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_url = request.url.host + request.url.path.replace('/', '_')
    # Limit filename length to avoid OS errors
    filename = f"{file_ts}_{request.method}_{safe_url[:50]}.log"
    filepath = os.path.join(HTTP_LOG_DIR, filename)
    
    # Decompress before formatting
    req_body_clean = _decompress_for_logs(req_body, request.headers)
    req_text = _format_body(req_body_clean)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("========== [MDC CONTEXT] ==========\n")
        current_mdc = mdc_context.get()
        if current_mdc:
            f.write(json.dumps(current_mdc, indent=2, ensure_ascii=False) + "\n\n")
        else:
            f.write("(Empty)\n\n")
        
        f.write(f"========== [HTTP REQUEST] ==========\n")
        f.write(f"URL: {request.method} {request.url}\n")
        f.write(f"Headers:\n{_format_headers(request.headers)}\n\n")
        f.write(f"Body:\n{req_text}\n\n")

        if exception:
            f.write("========== [EXCEPTION / ERROR] ==========\n")
            f.write(f"Type: {type(exception).__name__}\n")
            f.write("Stacktrace:\n")
            f.write("".join(traceback.format_exception(type(exception), exception, exception.__traceback__)))
        
        if response:
            # If the exception happened mid-stream, we might have partial response headers/body
            f.write("\n\n--- PARTIAL RESPONSE BEFORE EXCEPTION ---\n" if exception
                    else "========== [HTTP RESPONSE] ==========\n")
            status_marker = (" [ERROR: FAILED MID-STREAM]" if exception 
                             else " [ERROR: NON-2xx]" if is_error 
                             else "")
            f.write(f"Status: {response.status_code}{status_marker}\n")
            f.write(f"Headers:\n{_format_headers(response.headers)}\n\n")
            res_body_clean = _decompress_for_logs(res_body, response.headers)
            f.write(f"Body:\n{_format_body(res_body_clean)}\n")

def _decompress_for_logs(content_bytes: bytes, headers: httpx.Headers) -> bytes:
    """Decompresses network bytes for logging. Libraries are guaranteed to be present 
    if the server used these encodings."""
    if not content_bytes:
        return content_bytes
        
    encoding = headers.get("content-encoding", "").lower()
    
    try:
        if "zstd" in encoding:
            import zstandard
            dctx = zstandard.ZstdDecompressor()
            return dctx.decompress(content_bytes)
        elif "gzip" in encoding:
            import gzip
            return gzip.decompress(content_bytes)
        elif "deflate" in encoding:
            import zlib
            return zlib.decompress(content_bytes)
        elif "br" in encoding:
            import brotli
            return brotli.decompress(content_bytes)
    except Exception as e:
        # If it fails, just return the raw bytes so the app doesn't crash
        print(f"Failed to decompress {encoding} for logging: {e}")
        
    return content_bytes


def _format_body(content_bytes: bytes) -> str:
    """Attempts to format bytes as JSON, NDJSON (JSON Lines), or falls back to text."""
    if not content_bytes:
        return ""
    
    text = content_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return ""

    # 1. Try standard JSON
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
        
    # 2. Try JSON Lines (NDJSON) which is very common in LLM streaming (like Ollama)
    lines = text.split('\n')
    if len(lines) > 1:
        try:
            parsed_lines = [json.loads(line) for line in lines if line.strip()]
            return "\n\n".join(json.dumps(p, indent=2, ensure_ascii=False) for p in parsed_lines)
        except json.JSONDecodeError:
            pass
            
    # 3. Fallback to raw text
    return text

def _format_headers(headers: httpx.Headers) -> str:
    """Formats headers as a neat, multiline string."""
    if not headers:
        return "  (None)"
    return "\n".join(f"  {k}: {v}" for k, v in headers.items())

###############################################################################
#                                ASYNC                                        #
###############################################################################

class FileLoggingAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, request: httpx.Request, req_body: bytes, response: httpx.Response):
        self.stream = stream
        self.request = request
        self.req_body = req_body
        self.response = response
        self.chunks = []
        self._logged = False

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                self.chunks.append(chunk)
                yield chunk
        except Exception as e:
            # Catch exceptions that occur DURING streaming (e.g., ReadTimeout)
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e)
                self._logged = True
            raise e

    async def aclose(self):
        try:
            await self.stream.aclose()
        except Exception as e:
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e)
                self._logged = True
            raise e

        if not self._logged:
            res_body = b"".join(self.chunks)
            _write_log_file(self.request, self.req_body, self.response, res_body)
            self._logged = True


class FileLoggingAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, delegate: httpx.AsyncBaseTransport):
        super().__init__()
        self.delegate = delegate

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            await request.aread()
            req_body = request.content
        except Exception as e:
            # If reading the outgoing body fails, log what we have (URL/Headers)
            _write_log_file(request, b"<FAILED TO READ REQUEST BODY>", exception=e)
            raise e
        
        try:
            response = await self.delegate.handle_async_request(request)
        except Exception as e:
            # Catch exceptions that occur BEFORE the response is received (e.g., ConnectionError)
            _write_log_file(request, req_body, exception=e)
            raise e
        
        response.stream = FileLoggingAsyncByteStream(
            response.stream, request, req_body, response
        )
        return response

    async def aclose(self) -> None:
        await self.delegate.aclose()

###############################################################################
#                                 SYNC                                        #
###############################################################################

class FileLoggingSyncByteStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, request: httpx.Request, req_body: bytes, response: httpx.Response):
        self.stream = stream
        self.request = request
        self.req_body = req_body
        self.response = response
        self.chunks = []
        self._logged = False

    def __iter__(self):
        try:
            for chunk in self.stream:
                self.chunks.append(chunk)
                yield chunk
        except Exception as e:
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e)
                self._logged = True
            raise e

    def close(self):
        try:
            self.stream.close()
        except Exception as e:
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e)
                self._logged = True
            raise e

        if not self._logged:
            res_body = b"".join(self.chunks)
            _write_log_file(self.request, self.req_body, self.response, res_body)
            self._logged = True


class FileLoggingSyncTransport(httpx.BaseTransport):
    def __init__(self, delegate: httpx.BaseTransport):
        super().__init__()
        self.delegate = delegate

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            request.read()
            req_body = request.content
        except Exception as e:
            # If reading the outgoing body fails, log what we have (URL/Headers)
            _write_log_file(request, b"<FAILED TO READ REQUEST BODY>", exception=e)
            raise e

        try:
            response = self.delegate.handle_request(request)
        except Exception as e:
            _write_log_file(request, req_body, exception=e)
            raise e
        
        response.stream = FileLoggingSyncByteStream(
            response.stream, request, req_body, response
        )
        return response

    def close(self) -> None:
        self.delegate.close()