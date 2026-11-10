"""Start Olive Table locally and open its page once the server is ready.

This launcher uses only the standard library until it imports the application.
It never stops an existing process or replaces a service using the chosen port.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


HOST = "127.0.0.1"
DEFAULT_PORT = 8765
READINESS_TIMEOUT_SECONDS = 60.0

# A local readiness check should not go through a machine's HTTP proxy.
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be a number from 1 to 65535") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be a number from 1 to 65535")
    return port


def _port_is_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex((HOST, port)) == 0


def _is_olive_table(url: str) -> bool:
    """Recognise the application rather than reusing any HTTP service."""
    try:
        with LOCAL_HTTP.open(url + "/api/health", timeout=1.5) as response:
            if response.status != 200:
                return False
            payload = response.read(65537)
        if len(payload) > 65536:
            return False
        health = json.loads(payload)
        return (
            isinstance(health, dict)
            and health.get("status") == "ok"
            and health.get("engine") == "offline-nlp"
        )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _open_page(url: str) -> None:
    try:
        opened = webbrowser.open(url, new=2)
    except Exception:
        opened = False
    if not opened:
        print(f"Open this address in your browser: {url}", flush=True)


def _wait_until_ready(url: str, stop: threading.Event, open_browser: bool) -> None:
    deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
    while not stop.is_set() and time.monotonic() < deadline:
        if _is_olive_table(url):
            print(f"Olive Table is ready: {url}", flush=True)
            if open_browser and not stop.is_set():
                _open_page(url)
            return
        if stop.wait(0.25):
            return
    if not stop.is_set():
        print(
            f"The page did not become ready within 60 seconds. "
            f"Check the server messages, then try {url} in your browser.",
            file=sys.stderr,
            flush=True,
        )


def check_installation() -> int:
    """Check the real model and API in process without opening a network port."""
    try:
        from app import create_app

        client = create_app(testing=True).test_client()
        health_response = client.get("/api/health")
        health = health_response.get_json()
        if health_response.status_code != 200 or not isinstance(health, dict):
            raise RuntimeError("the health API did not respond correctly")
        if health.get("engine") != "offline-nlp" or health.get("status") != "ok":
            raise RuntimeError("the expected Olive Table engine did not load")
        if health.get("trained_model_loaded") is not True:
            raise RuntimeError("the trained model is missing or could not load")

        config_response = client.get("/api/config")
        config = config_response.get_json()
        if config_response.status_code != 200 or not isinstance(config, dict):
            raise RuntimeError("the menu API did not respond correctly")
        menu = config.get("menu")
        if not isinstance(menu, list) or len(menu) != 36:
            raise RuntimeError("the complete 36-item menu did not load")

        chat_response = client.post(
            "/api/chat", json={"message": "What are your opening hours?"}
        )
        chat = chat_response.get_json()
        if (
            chat_response.status_code != 200
            or not isinstance(chat, dict)
            or not isinstance(chat.get("reply"), str)
            or not chat["reply"].strip()
            or not chat.get("session_id")
            or not chat.get("sources")
        ):
            raise RuntimeError("the conversation API did not return a grounded answer")
    except Exception as exc:
        print(f"Olive Table setup check failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print("Olive Table check passed: trained model, 36 menu items and chat API.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start Olive Table and open its local web page.")
    parser.add_argument("--port", type=_port_number, default=DEFAULT_PORT, help="local port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="start without opening a browser")
    parser.add_argument("--check", action="store_true", help="check the model, menu and API without starting a server")
    args = parser.parse_args(argv)
    if args.check:
        return check_installation()

    url = f"http://{HOST}:{args.port}"
    if _port_is_in_use(args.port):
        if _is_olive_table(url):
            print(f"Olive Table is already running: {url}", flush=True)
            if not args.no_browser:
                _open_page(url)
            return 0
        print(
            f"Port {args.port} is already in use by another application. "
            f"Choose another port, for example: python launch.py --port {args.port + 1 if args.port < 65535 else 8765}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    try:
        from app import create_app

        application = create_app()
    except Exception as exc:
        print(f"Olive Table could not start: {exc}", file=sys.stderr, flush=True)
        print("Run START_WINDOWS.bat again to check the local installation.", file=sys.stderr, flush=True)
        return 1

    print(f"Starting Olive Table at {url}. Keep this window open while using the app.", flush=True)
    stop = threading.Event()
    readiness = threading.Thread(
        target=_wait_until_ready,
        args=(url, stop, not args.no_browser),
        name="olive-table-browser-readiness",
        daemon=True,
    )
    readiness.start()
    try:
        application.run(host=HOST, port=args.port, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\nOlive Table stopped.", flush=True)
    except OSError as exc:
        print(f"Olive Table could not use {url}: {exc}", file=sys.stderr, flush=True)
        return 2
    except SystemExit as exc:
        # Flask can exit when another service takes the port between the probe
        # and bind. Report the failure without touching that other process.
        code = exc.code if isinstance(exc.code, int) else 1
        if code:
            print(f"Olive Table could not start at {url}. Check the server message above.", file=sys.stderr, flush=True)
            return code
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
