"""HTTP/JSON validation and isolated conversation behavior through Flask's client."""

import json
import sys
import unittest
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(testing=True)
        self.client = self.app.test_client()

    def chat(self, message, session_id=None, mode="enhanced"):
        body = {"message": message, "mode": mode}
        if session_id is not None:
            body["session_id"] = session_id
        response = self.client.post("/api/chat", json=body)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        for key in ("reply", "intent", "session_id", "context", "sources"):
            self.assertIn(key, payload)
        uuid.UUID(payload["session_id"])
        return payload

    def assert_json_error(self, response, expected_status=400):
        self.assertEqual(response.status_code, expected_status, response.get_data(as_text=True))
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIsInstance(payload.get("error"), dict)
        self.assertIsInstance(payload["error"].get("code"), str)
        self.assertTrue(payload["error"].get("message"))

    def test_health_and_public_configuration_are_json(self):
        for endpoint in ("/api/health", "/api/config"):
            with self.subTest(endpoint=endpoint):
                response = self.client.get(endpoint)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.is_json)
                self.assertIsInstance(response.get_json(), dict)

    def test_missing_empty_and_wrong_type_messages_are_rejected(self):
        invalid_bodies = [
            {}, {"message": ""}, {"message": " \r\n\t "},
            {"message": None}, {"message": 123}, {"message": True},
            {"message": ["hello"]}, {"message": {"text": "hello"}},
        ]
        for body in invalid_bodies:
            with self.subTest(body=body):
                self.assert_json_error(self.client.post("/api/chat", json=body))

    def test_malformed_json_and_non_object_json_are_rejected(self):
        for raw in ('{"message":', '["hello"]', '"hello"', '123', 'null'):
            with self.subTest(raw=raw):
                self.assert_json_error(self.client.post("/api/chat", data=raw, content_type="application/json"))

    def test_invalid_mode_session_id_and_oversize_message_are_rejected(self):
        for body in (
            {"message": "Hello", "mode": "invented-mode"},
            {"message": "Hello", "mode": ["enhanced"]},
            {"message": "Hello", "session_id": "../another-user"},
            {"message": "Hello", "session_id": 42},
            {"message": "x" * 1201},
        ):
            with self.subTest(body=body):
                self.assert_json_error(self.client.post("/api/chat", json=body))

    def test_session_preferences_survive_followup_and_are_isolated(self):
        vegan = self.chat("I am vegan and my budget is 14 pounds.")
        second = self.chat("Hello there.")
        self.assertNotEqual(vegan["session_id"], second["session_id"])
        self.assertIsNone(second["context"]["diet"])
        followup = self.chat("Please suggest a main course.", vegan["session_id"])
        self.assertEqual(followup["session_id"], vegan["session_id"])
        self.assertEqual(followup["context"]["diet"], "vegan")
        self.assertEqual(followup["context"]["max_price"], 14)
        isolated = self.chat("Please suggest a main course.", second["session_id"])
        self.assertIsNone(isolated["context"]["diet"])
        self.assertIsNone(isolated["context"]["max_price"])

    def test_deleting_session_clears_preferences_and_unknown_delete_is_404(self):
        original = self.chat("I am vegetarian and my budget is 15 pounds.")
        response = self.client.delete("/api/sessions/" + original["session_id"])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        self.assertTrue(response.get_json()["ok"])
        restarted = self.chat("Hello there.", original["session_id"])
        self.assertIsNone(restarted["context"]["diet"])
        self.assertIsNone(restarted["context"]["max_price"])
        self.assert_json_error(self.client.delete("/api/sessions/" + str(uuid.uuid4())), 404)

    def test_html_shaped_message_remains_data_and_cannot_change_next_answer(self):
        malicious = '<script>alert("x")</script>\n\\\"</textarea><img src=x onerror=alert(1)>'
        first = self.chat(malicious)
        self.assertEqual(first["intent"], "fallback")
        self.assertEqual(first["sources"], [])
        self.assertNotIn("<script", first["reply"].lower())
        self.assertNotIn("onerror=", first["reply"].lower())
        normal = self.chat("What are your opening hours?", first["session_id"])
        self.assertEqual(normal["intent"], "hours")
        # Literal quotes and backslashes must still produce valid JSON, not markup.
        self.assertEqual(json.loads(json.dumps(first)), first)


if __name__ == "__main__":
    unittest.main()
