"""Local Flask interface for the Olive Table coursework prototype."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from collections import OrderedDict

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from restaurant_bot import RestaurantBot


class SessionStore:
    """Bounded, expiring in-memory state; no raw chat transcript is stored.

    The lock protects read-modify-write dialogue turns. Sessions disappear on
    restart, which is intentional for a local classroom demo rather than a
    production reservation system.
    """

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.lock = threading.RLock()

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [key for key, (stamp, _) in self.sessions.items() if now - stamp > self.ttl_seconds]
        for key in expired:
            self.sessions.pop(key, None)

    def get_or_create(self, supplied_id: str | None) -> tuple[str, dict]:
        self._prune()
        if supplied_id in self.sessions:
            _, state = self.sessions.pop(supplied_id)
            self.sessions[supplied_id] = (time.monotonic(), state)
            return supplied_id, state
        while len(self.sessions) >= self.max_sessions:
            self.sessions.popitem(last=False)
        session_id = str(uuid.uuid4())
        state = RestaurantBot.new_state()
        self.sessions[session_id] = (time.monotonic(), state)
        return session_id, state


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config.update(TESTING=testing, MAX_CONTENT_LENGTH=16 * 1024)
    bot = RestaurantBot()
    store = SessionStore()
    app.extensions["restaurant_bot"] = bot
    app.extensions["dialogue_sessions"] = store

    def error(code: str, message: str, status: int = 400):
        return jsonify({"error": {"code": code, "message": message}}), status

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith("/api/"):
            return error("request_too_large" if exc.code == 413 else "http_error", exc.description, exc.code)
        return exc

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "engine": "offline-nlp", "version": "1.0.0", "trained_model_loaded": bot.classifier is not None})

    @app.get("/api/config")
    def config():
        return jsonify(copy.deepcopy(bot.config))

    @app.post("/api/chat")
    def chat():
        if not request.is_json:
            return error("invalid_json", "Send a JSON object with a message field.")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return error("invalid_json", "The request body must be a valid JSON object.")
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            return error("invalid_message", "Enter a non-empty text message.")
        if len(message) > 1200:
            return error("message_too_long", "Keep your message to 1200 characters or fewer.")
        mode = payload.get("mode", "enhanced")
        if not isinstance(mode, str) or mode not in {"baseline", "enhanced"}:
            return error("invalid_mode", "Choose enhanced or baseline mode.")
        supplied_id = payload.get("session_id")
        if supplied_id is not None:
            if not isinstance(supplied_id, str):
                return error("invalid_session", "The session ID must be a UUID string.")
            try:
                if str(uuid.UUID(supplied_id)) != supplied_id:
                    raise ValueError("non-canonical UUID")
            except (ValueError, AttributeError):
                return error("invalid_session", "The session ID must be a valid UUID string.")
        start = time.perf_counter()
        with store.lock:
            session_id, state = store.get_or_create(supplied_id)
            result = bot.reply(message.strip(), state, mode)
        result["session_id"] = session_id
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return jsonify(result)

    @app.delete("/api/sessions/<session_id>")
    def reset_session(session_id):
        with store.lock:
            store._prune()
            if store.sessions.pop(session_id, None) is None:
                return error("session_not_found", "That conversation has already expired or been cleared.", 404)
        return jsonify({"ok": True})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8765, debug=False, threaded=True)
