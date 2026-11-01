import os
import json
import time
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
#                              TIMINGS                                        #
###############################################################################

class _Timings:
    def __init__(self):
        self.t1_before_req_read: float = 0.0
        self.t2_before_network: float = 0.0
        self.t3_start_stream: float = 0.0
        self.t4_start_close: float = 0.0
        self.t5_log_write: float = 0.0

    def format_log_section(self) -> str:
        lines = ["========== [TIMINGS] =========="]
        
        def fmt_ts(ts):
            return datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3] if ts else "(Not Reached)"
            
        lines.append(f"1. Before Request Read : {fmt_ts(self.t1_before_req_read)}")
        lines.append(f"2. Before Network Req  : {fmt_ts(self.t2_before_network)}")
        lines.append(f"3. Start of Stream     : {fmt_ts(self.t3_start_stream)}")
        lines.append(f"4. Start of Close      : {fmt_ts(self.t4_start_close)}")
        lines.append(f"5. Log File Written    : {fmt_ts(self.t5_log_write)}")
        
        lines.append("\n--- Durations ---")
        
        def fmt_diff(t_end, t_start):
            return f"{(t_end - t_start):.3f}s" if t_end and t_start else "N/A"
            
        lines.append(f"Local Body Read        : {fmt_diff(self.t2_before_network, self.t1_before_req_read)}")
        lines.append(f"Time to First Byte     : {fmt_diff(self.t3_start_stream, self.t2_before_network)}")
        lines.append(f"Stream Consumption     : {fmt_diff(self.t4_start_close, self.t3_start_stream)}")
        lines.append(f"Total Network Time     : {fmt_diff(self.t4_start_close, self.t2_before_network)}")
        
        return "\n".join(lines) + "\n\n"
    
###############################################################################
#                              LOGGING                                        #
###############################################################################


def _write_log_file(
        request: httpx.Request, 
        req_body: bytes, 
        response: httpx.Response = None, 
        res_body: bytes = b"", 
        exception: Exception = None,
        timings: _Timings = None,
):
    """Writes the formatted request and response pair to a new timestamped file."""
    if not HTTP_LOG_DIR: 
        return

    # Record the final file write timing
    if timings:
        timings.t5_log_write = time.time()

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

        if timings:
            f.write(timings.format_log_section())
        
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
    def __init__(self, stream: httpx.AsyncByteStream, request: httpx.Request, req_body: bytes, response: httpx.Response, timings: _Timings):
        self.stream = stream
        self.request = request
        self.req_body = req_body
        self.response = response
        self.timings = timings
        self.chunks = []
        self._logged = False

    async def __aiter__(self):
        if not self.timings.t3_start_stream:
            self.timings.t3_start_stream = time.time()

        try:
            async for chunk in self.stream:
                self.chunks.append(chunk)
                yield chunk
        except Exception as e:
            # Catch exceptions that occur DURING streaming (e.g., ReadTimeout)
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e, timings=self.timings)
                self._logged = True
            raise e

    async def aclose(self):
        if not self.timings.t4_start_close:
            self.timings.t4_start_close = time.time()

        try:
            await self.stream.aclose()
        except Exception as e:
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e, timings=self.timings)
                self._logged = True
            raise e

        if not self._logged:
            res_body = b"".join(self.chunks)
            _write_log_file(self.request, self.req_body, self.response, res_body, timings=self.timings)
            self._logged = True


class FileLoggingAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, delegate: httpx.AsyncBaseTransport):
        super().__init__()
        self.delegate = delegate

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        timings = _Timings()

        try:
            timings.t1_before_req_read = time.time()
            await request.aread()
            req_body = request.content
        except Exception as e:
            # If reading the outgoing body fails, log what we have (URL/Headers)
            _write_log_file(request, b"<FAILED TO READ REQUEST BODY>", exception=e, timings=timings)
            raise e
        
        try:
            timings.t2_before_network = time.time()
            response = await self.delegate.handle_async_request(request)
        except Exception as e:
            # Catch exceptions that occur BEFORE the response is received (e.g., ConnectionError)
            _write_log_file(request, req_body, exception=e, timings=timings)
            raise e
        
        response.stream = FileLoggingAsyncByteStream(
            response.stream, request, req_body, response, timings
        )
        return response

    async def aclose(self) -> None:
        await self.delegate.aclose()

###############################################################################
#                                 SYNC                                        #
###############################################################################

class FileLoggingSyncByteStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, request: httpx.Request, req_body: bytes, response: httpx.Response, timings: _Timings):
        self.stream = stream
        self.request = request
        self.req_body = req_body
        self.response = response
        self.timings = timings
        self.chunks = []
        self._logged = False

    def __iter__(self):
        if not self.timings.t3_start_stream:
            self.timings.t3_start_stream = time.time()

        try:
            for chunk in self.stream:
                self.chunks.append(chunk)
                yield chunk
        except Exception as e:
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e, timings=self.timings)
                self._logged = True
            raise e

    def close(self):
        if not self.timings.t4_start_close:
            self.timings.t4_start_close = time.time()

        try:
            self.stream.close()
        except Exception as e:
            if not self._logged:
                res_body = b"".join(self.chunks)
                _write_log_file(self.request, self.req_body, self.response, res_body, exception=e, timings=self.timings)
                self._logged = True
            raise e

        if not self._logged:
            res_body = b"".join(self.chunks)
            _write_log_file(self.request, self.req_body, self.response, res_body, timings=self.timings)
            self._logged = True


class FileLoggingSyncTransport(httpx.BaseTransport):
    def __init__(self, delegate: httpx.BaseTransport):
        super().__init__()
        self.delegate = delegate

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        timings = _Timings()

        try:
            timings.t1_before_req_read = time.time()
            request.read()
            req_body = request.content
        except Exception as e:
            # If reading the outgoing body fails, log what we have (URL/Headers)
            _write_log_file(request, b"<FAILED TO READ REQUEST BODY>", exception=e, timings=timings)
            raise e

        try:
            timings.t2_before_network = time.time()
            response = self.delegate.handle_request(request)
        except Exception as e:
            _write_log_file(request, req_body, exception=e, timings=timings)
            raise e
        
        response.stream = FileLoggingSyncByteStream(
            response.stream, request, req_body, response, timings
        )
        return response

    def close(self) -> None:
        self.delegate.close()