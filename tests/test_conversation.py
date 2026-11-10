"""User-level dialogue contracts, with local authored menu data as the oracle."""

import json
import math
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from restaurant_bot import RestaurantBot


class ConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads((PROJECT / "data" / "restaurant.json").read_text(encoding="utf-8"))
        cls.menu = {dish["id"]: dish for dish in data["menu"]}

    def setUp(self):
        self.bot = RestaurantBot()

    def say(self, message, state=None):
        answer = self.bot.reply(message, state=state, mode="enhanced")
        for key in ("reply", "intent", "confidence", "sources", "context", "suggested_prompts"):
            self.assertIn(key, answer)
        self.assertIsInstance(answer["reply"], str)
        self.assertTrue(answer["reply"].strip())
        self.assertTrue(math.isfinite(answer["confidence"]))
        self.assertGreaterEqual(answer["confidence"], 0)
        self.assertLessEqual(answer["confidence"], 1)
        # The return value must travel through the JSON API without custom encoders.
        self.assertEqual(json.loads(json.dumps(answer)), answer)
        return answer

    def recommended(self, answer):
        ids = answer["context"]["recommended_ids"]
        self.assertTrue(ids, "The feasible authored menu contains matching dishes")
        self.assertEqual(len(set(ids)), len(ids))
        for dish_id in ids:
            self.assertIn(dish_id, self.menu)
        return [self.menu[dish_id] for dish_id in ids]

    def test_vegetarian_can_be_corrected_to_vegan_across_turns(self):
        state = {}
        first = self.say("Please recommend a vegetarian main course.", state)
        self.assertEqual(first["context"]["diet"], "vegetarian")
        for dish in self.recommended(first):
            self.assertIn("vegetarian", dish["tags"])
            self.assertEqual(dish["category"], "main")
        corrected = self.say("Actually, I am vegan.", state)
        self.assertEqual(corrected["context"]["diet"], "vegan")
        for dish in self.recommended(corrected):
            self.assertIn("vegan", dish["tags"])
            self.assertEqual(dish["category"], "main")

    def test_followup_budget_preserves_diet_and_filters_actual_prices(self):
        state = {}
        self.say("I would like a vegan main course.", state)
        budget = self.say("My budget is 14 pounds.", state)
        self.assertEqual(budget["context"]["diet"], "vegan")
        self.assertEqual(budget["context"]["max_price"], 14)
        for dish in self.recommended(budget):
            self.assertIn("vegan", dish["tags"])
            self.assertLessEqual(dish["price"], 14)
            self.assertEqual(dish["category"], "main")

    def test_impossible_budget_is_explained_without_overbudget_recommendations(self):
        state = {}
        result = self.say("Recommend a vegan main for under 1 pound.", state)
        self.assertEqual(result["context"]["max_price"], 1)
        self.assertEqual(result["context"]["recommended_ids"], [])
        self.assertTrue(any(word in result["reply"].lower() for word in ("no ", "couldn't", "cannot", "could not", "can't", "match")))

    def test_named_dish_supports_price_followup(self):
        state = {}
        first = self.say("Tell me about the Garden lentil bowl.", state)
        self.assertEqual(first["intent"], "dish_details")
        self.assertEqual(first["context"]["selected_dish_id"], "lentil-bowl")
        followup = self.say("How much is it?", state)
        self.assertEqual(followup["context"]["selected_dish_id"], "lentil-bowl")
        self.assertIn("13", followup["reply"])

    def test_unrelated_question_falls_back_without_erasing_preferences(self):
        state = {}
        self.say("I want a vegetarian main under 15 pounds.", state)
        before = {key: state.get(key) for key in ("diet", "max_price", "category")}
        result = self.say("Explain how nuclear fusion works inside a star.", state)
        self.assertEqual(result["intent"], "fallback")
        self.assertEqual(result["sources"], [])
        for key, value in before.items():
            self.assertEqual(result["context"].get(key), value)

    def test_omitted_state_and_separate_state_do_not_share_preferences(self):
        first_state, second_state = {}, {}
        self.say("I am vegan and my budget is 14 pounds.", first_state)
        separate = self.say("Hello there.", second_state)
        fresh = self.say("Hello there.")
        for result in (separate, fresh):
            self.assertIsNone(result["context"]["diet"])
            self.assertIsNone(result["context"]["max_price"])
        self.assertEqual(first_state["diet"], "vegan")

    def test_reset_clears_previous_constraints(self):
        state = {}
        self.say("I am vegan and my budget is 14 pounds.", state)
        reset = self.say("reset", state)
        self.assertEqual(reset["intent"], "reset")
        self.assertIsNone(reset["context"]["diet"])
        self.assertIsNone(reset["context"]["max_price"])
        self.assertEqual(reset["context"]["recommended_ids"], [])

    def test_baseline_and_enhanced_produce_source_backed_faq_answers(self):
        for mode in ("baseline", "enhanced"):
            with self.subTest(mode=mode):
                result = self.bot.reply("What are your opening hours?", mode=mode)
                self.assertEqual(result["intent"], "hours")
                self.assertTrue(result["sources"])
                self.assertIn("id", result["sources"][0])
                self.assertIn("title", result["sources"][0])


if __name__ == "__main__":
    unittest.main()
