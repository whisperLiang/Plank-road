from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JsonHandler = Callable[[dict[str, Any]], dict[str, Any]]


class JsonRpcError(RuntimeError):
    pass


def encode_bytes(value: bytes | bytearray | None) -> str:
    return base64.b64encode(bytes(value or b"")).decode("ascii")


def decode_bytes(value: object) -> bytes:
    if value in (None, ""):
        return b""
    return base64.b64decode(str(value).encode("ascii"))


def post_json(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    url = f"http://{endpoint}{path}"
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(timeout)) as response:
            response_data = response.read()
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace") or str(exc)
        raise JsonRpcError(message) from exc
    except URLError as exc:
        raise JsonRpcError(str(exc)) from exc
    if not response_data:
        return {}
    return json.loads(response_data.decode("utf-8"))


class JsonRpcServer:
    def __init__(
        self,
        *,
        listen_address: str,
        routes: dict[str, JsonHandler],
        health_payload: dict[str, Any] | None = None,
    ) -> None:
        host, port_text = listen_address.rsplit(":", 1)
        self.routes = dict(routes)
        self.httpd = _build_http_server(
            host,
            int(port_text),
            self.routes,
            health_payload=dict(health_payload or {}),
        )
        actual_host, actual_port = self.httpd.server_address
        self.listen_address = f"{actual_host}:{actual_port}"
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.httpd.serve_forever,
            name=f"json-rpc-{self.listen_address}",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _build_http_server(
    host: str,
    port: int,
    routes: dict[str, JsonHandler],
    *,
    health_payload: dict[str, Any],
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"ok": True, **health_payload})
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:
            handler = routes.get(self.path)
            if handler is None:
                self.send_error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8") or "{}")
                result = handler(dict(payload or {}))
            except Exception as exc:
                self._send_json({"success": False, "message": str(exc)}, status=500)
                return
            self._send_json(result)

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, int(port)), Handler)
