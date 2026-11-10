"""Optional smoke check: real localhost HTTP plus piped command-line interaction."""

import json
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app import create_app
from werkzeug.serving import make_server


def main():
    server = make_server("127.0.0.1", 0, create_app(testing=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=10) as response:
            page = response.read().decode("utf-8")
            assert response.status == 200 and "Olive Table" in page
            assert response.headers["Content-Security-Policy"]
        with urllib.request.urlopen(base + "/api/health", timeout=10) as response:
            health = json.load(response)
            assert response.status == 200 and health["status"] == "ok"
        request = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps({"message": "Can I pay by credit card?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            answer = json.load(response)
            assert response.status == 200 and answer["intent"] == "payment", answer
            assert answer["sources"] and answer["session_id"]
        print("Real localhost HTTP: page, health and source-backed chat passed.")
        print("Trained model loaded:", health["trained_model_loaded"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    cli = subprocess.run(
        [sys.executable, str(PROJECT / "cli.py"), "--show-sources"],
        input="I am vegan.\nMy budget is 14 pounds.\n/reset\nWhat are your opening hours?\n/quit\n",
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=PROJECT, timeout=45,
    )
    assert cli.returncode == 0, cli.stderr
    assert "Intent: recommendation" in cli.stdout, cli.stdout
    assert "conversation has been reset" in cli.stdout, cli.stdout
    assert "Intent: hours" in cli.stdout and "Goodbye!" in cli.stdout, cli.stdout
    print("Command-line interface: dietary/budget turns, reset, sourced FAQ and quit passed.")


if __name__ == "__main__":
    main()
