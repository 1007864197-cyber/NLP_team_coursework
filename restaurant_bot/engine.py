"""Inspectable NLP pipeline: TF-IDF retrieval plus explicit dialogue state.

The indexed examples are authored restaurant FAQs, not an external training set.
Confidence values are cosine similarity scores, not calibrated probabilities.
No messages, secrets or external network calls are required by this module.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_STATE = {
    "diet": None,
    "excluded_ingredients": [],
    "max_price": None,
    "spice": None,
    "category": None,
    "dish_kind": None,
    "caffeine": None,
    "recommended_ids": [],
    "selected_dish_id": None,
    "last_intent": None,
}

ALLERGY_NOTE = (
    "These suggestions use listed ingredients only. Our shared kitchen means I "
    "cannot guarantee freedom from cross-contact; please confirm allergies with "
    "the restaurant team before ordering."
)


def normalise(text: str) -> str:
    """Normalise punctuation/case without discarding negation or currency."""
    text = text.lower().replace("’", "'").replace("–", "-")
    text = re.sub(r"\bgluten[- ]?free\b", "gluten free", text)
    text = re.sub(r"\b(?:vegeterian|vegatarian)\b", "vegetarian", text)
    return re.sub(r"\s+", " ", text).strip()


class RestaurantBot:
    """Stateless model object; conversation state belongs to its caller.

    ``reply(message, state, mode)`` updates a supplied mutable state dictionary.
    Reuse it for follow-up turns; pass ``{}`` or omit it for a new conversation.
    The model's TF-IDF matrices are fitted once and never updated from chat text.
    """

    def __init__(self, data_dir: str | Path | None = None):
        base = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
        self.config = json.loads((base / "restaurant.json").read_text(encoding="utf-8"))
        self.faqs = json.loads((base / "faq.json").read_text(encoding="utf-8"))
        self.menu = self.config["menu"]
        self.dishes = {item["id"]: item for item in self.menu}
        self.faq_by_intent = {item["intent"]: item for item in self.faqs}
        examples = [(normalise(example), faq) for faq in self.faqs for example in faq["examples"]]
        self.examples = [example for example, _ in examples]
        self.example_faqs = [faq for _, faq in examples]

        # Baseline: word unigrams. Enhanced retrieval adds local word order and
        # character fragments, which help with spelling variation and inflection.
        self.baseline_vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.baseline_matrix = self.baseline_vectorizer.fit_transform(self.examples)
        self.word_matrix = self.word_vectorizer.fit_transform(self.examples)
        self.char_matrix = self.char_vectorizer.fit_transform(self.examples)
        # Training is a separate, reproducible command. Only a locally fitted
        # model is loaded; starting from source without it keeps the TF-IDF demo
        # usable. The response method always reveals which path was used.
        self.classifier = None
        try:
            from .classifier import IntentClassifier
            self.classifier = IntentClassifier()
        except (ImportError, FileNotFoundError):
            pass

    def retrieve(self, message: str, mode: str = "enhanced") -> dict[str, Any]:
        """Classify a single FAQ query with fixed, pre-evaluation thresholds.

        A low top score or weak evidence yields fallback. No ground-truth labels
        from the evaluation file are read here or used to fit these vectorizers.
        """
        if mode not in {"baseline", "enhanced"}:
            raise ValueError("mode must be 'baseline' or 'enhanced'")
        query = normalise(message)
        if mode == "baseline":
            scores = cosine_similarity(self.baseline_vectorizer.transform([query]), self.baseline_matrix)[0]
            threshold = 0.23
        else:
            words = cosine_similarity(self.word_vectorizer.transform([query]), self.word_matrix)[0]
            chars = cosine_similarity(self.char_vectorizer.transform([query]), self.char_matrix)[0]
            scores = 0.72 * words + 0.28 * chars
            threshold = 0.20
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        faq = self.example_faqs[best_index]
        # Character overlap by itself can spuriously match unrelated words.
        # Require one actual content token shared with the winning FAQ examples.
        stop = {"a", "an", "and", "are", "at", "be", "by", "can", "do", "does", "for", "from", "have", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "the", "there", "this", "to", "we", "what", "when", "where", "which", "with", "you", "your"}
        query_tokens = set(re.findall(r"[a-z]+", query)) - stop
        evidence_tokens = set(re.findall(r"[a-z]+", " ".join(faq["examples"]).lower())) - stop
        overlap = query_tokens & evidence_tokens
        if best_score < threshold or (not overlap and best_score < 0.42):
            return {"intent": "fallback", "confidence": round(best_score, 4), "source_id": None}
        return {"intent": faq["intent"], "confidence": round(best_score, 4), "source_id": faq["id"]}

    @staticmethod
    def new_state() -> dict[str, Any]:
        return copy.deepcopy(DEFAULT_STATE)

    def _ensure_state(self, state: dict[str, Any]) -> None:
        for key, value in DEFAULT_STATE.items():
            if key not in state:
                state[key] = copy.deepcopy(value)

    def _result(self, reply: str, intent: str, state: dict[str, Any], confidence: float = 1.0,
                sources: list[dict[str, str]] | None = None,
                prompts: list[str] | None = None,
                method: str = "Dialogue rules and structured menu constraints") -> dict[str, Any]:
        state["last_intent"] = intent
        return {
            "reply": reply,
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "method": method,
            "sources": sources or [],
            "suggested_prompts": prompts or ["What would you recommend?", "When are you open?", "How can I book a table?"],
            "context": copy.deepcopy(state),
        }

    @staticmethod
    def _source(item: dict[str, Any]) -> dict[str, str]:
        return {"id": item["id"], "title": item.get("title", item.get("name", "Menu"))}

    def _extract_preferences(self, text: str, state: dict[str, Any]) -> tuple[bool, bool]:
        """Extract a deliberately small set of explainable slots using rules.

        Negation is handled before positive diet terms. Restrictions persist
        until changed or reset, so a follow-up such as 'under 15' is meaningful.
        """
        changed = False
        allergy = bool(re.search(r"\ballerg(?:y|ies|ic)\b|\bcross[- ]?contamin", text))
        if re.search(r"\b(?:no dietary restrictions|no restrictions|anything is fine)\b", text):
            for key in ("diet", "excluded_ingredients", "spice", "caffeine", "dish_kind"):
                state[key] = copy.deepcopy(DEFAULT_STATE[key])
            changed = True
        if re.search(r"\b(?:not|no longer) (?:a )?(?:vegan|vegetarian|pescatarian)\b", text):
            state["diet"] = None
            changed = True
        else:
            if re.search(r"\bvegan\b|\bplant[- ]based\b|\bno animal products\b", text):
                state["diet"] = "vegan"
                changed = True
            elif re.search(r"\bvegetarian\b|\b(?:don'?t|do not|never) eat meat\b|\bmeat[- ]free\b|\bno meat\b", text):
                state["diet"] = "vegetarian"
                changed = True
            elif re.search(r"\bpescatarian\b", text):
                state["diet"] = "pescatarian"
                changed = True

        ingredient_aliases = {
            "nuts": r"nuts?|peanuts?|walnuts?|almonds?|cashews?|hazelnuts?|pistachios?",
            "milk": r"milk|dairy|lactose|cheese|butter|cream",
            "wheat": r"wheat|gluten",
            "egg": r"eggs?",
            "sesame": r"sesame|tahini",
            "fish": r"fish|salmon",
            "crustaceans": r"shellfish|crustaceans?|prawns?|shrimps?",
            "molluscs": r"molluscs?|squid|calamari",
            "mushroom": r"mushrooms?",
            "celery": r"celery",
            "mustard": r"mustard",
            "chicken": r"chicken",
        }
        for canonical, aliases in ingredient_aliases.items():
            # A bare ingredient question must not silently become an exclusion.
            excluded = bool(
                re.search(rf"\b(?:no|without|avoid|excluding|exclude|cannot eat|can'?t eat|don'?t eat|do not eat|don'?t like|do not like|allergic to|allergy to|intolerant to)\s+(?:(?:any|all)\s+)?(?:{aliases})\b", text)
                or re.search(rf"\b(?:{aliases})[- ]free\b", text)
                or (allergy and re.search(rf"\b(?:{aliases})\b", text))
            )
            if excluded:
                if canonical not in state["excluded_ingredients"]:
                    state["excluded_ingredients"].append(canonical)
                changed = True

        budget_match = re.search(r"(?:under|below|less than|at most|up to|maximum|max|budget(?: of| is)?)\s*(?:£|gbp\s*)?(\d+(?:\.\d{1,2})?)", text)
        if budget_match:
            state["max_price"] = max(0.0, min(float(budget_match.group(1)), 1000.0))
            changed = True
        elif re.search(r"\b(?:no budget|any price|remove (?:the )?budget|no price limit)\b", text):
            state["max_price"] = None
            changed = True
        elif re.search(r"\b(?:cheap|affordable|inexpensive)\b", text) and not re.search(r"\bcheapest\b", text):
            state["max_price"] = 15.0
            changed = True

        if re.search(r"\bmild\b|\bnot spicy\b|\bnon[- ]spicy\b|\bno (?:chilli|chili|spice)\b|\bdon'?t like spicy\b", text):
            state["spice"] = "mild"
            changed = True
        elif re.search(r"\b(?:spicy|hot|chilli|chili)\b", text):
            state["spice"] = "spicy"
            changed = True
        for category, terms in {
            "starter": r"\b(?:starters?|appeti[sz]ers?)\b",
            "main": r"\b(?:mains?|main courses?)\b",
            "dessert": r"\b(?:desserts?|something sweet|sweet course)\b",
            "side": r"\b(?:sides?|side dishes|accompaniments?)\b",
            "drink": r"\b(?:drinks?|beverages?|something to drink)\b",
        }.items():
            if re.search(terms, text):
                state["category"] = category
                state["dish_kind"] = None
                changed = True
        for kind, pattern, category in [
            ("pasta", r"\b(?:pasta|ravioli|linguine|tagliatelle)\b", "main"),
            ("coffee", r"\b(?:coffee|espresso|flat white)\b", "drink"),
            ("tea", r"\btea\b", "drink"),
            ("water", r"\b(?:still|sparkling) water\b", "drink"),
        ]:
            if re.search(pattern, text):
                state["dish_kind"] = kind
                state["category"] = category
                changed = True
        if re.search(r"\b(?:caffeine[- ]free|no caffeine|without caffeine|decaf)\b", text):
            state["caffeine"] = "caffeine-free"
            state["category"] = "drink"
            changed = True
        elif re.search(r"\b(?:any caffeine|caffeine is fine|remove caffeine restriction)\b", text):
            state["caffeine"] = None
            changed = True
        return changed, allergy

    def _dish_mentioned(self, text: str, state: dict[str, Any]) -> dict[str, Any] | None:
        # Long distinctive names outrank common ingredients such as tomato.
        aliases = {
            "lentil-bowl": ["lentil bowl", "garden lentil"],
            "mushroom-risotto": ["mushroom risotto", "risotto"],
            "lemon-chicken": ["lemon herb chicken", "lemon chicken"],
            "roasted-aubergine": ["roasted aubergine", "aubergine"],
            "tomato-pasta": ["tomato basil pasta", "tomato pasta"],
            "salmon": ["pan-seared salmon", "pan seared salmon", "salmon"],
            "chickpea-curry": ["chickpea coconut curry", "chickpea curry", "coconut curry"],
            "margherita-flatbread": ["margherita", "flatbread"],
            "tomato-soup": ["tomato soup"],
            "hummus-plate": ["hummus", "pita"],
            "lemon-tart": ["lemon tart"],
            "berry-sorbet": ["berry sorbet", "sorbet"],
            "chocolate-brownie": ["chocolate walnut brownie", "brownie"],
            "prawn-linguine": ["prawn linguine", "linguine"],
        }
        for item_id, names in aliases.items():
            if any(re.search(rf"\b{re.escape(name)}\b", text) for name in names):
                return self.dishes[item_id]
        for item in sorted(self.menu, key=lambda entry: len(entry["name"]), reverse=True):
            if normalise(item["name"]) in text or item["id"].replace("-", " ") in text:
                return item
        ordinals = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
        ids = state["recommended_ids"]
        for word, index in ordinals.items():
            if re.search(rf"\b{word}\b", text) and len(ids) > index:
                return self.dishes[ids[index]]
        if re.search(r"\b(?:it|that|this dish|that dish)\b", text):
            selected = state.get("selected_dish_id")
            if selected in self.dishes:
                return self.dishes[selected]
            if len(ids) == 1:
                return self.dishes[ids[0]]
        return None

    def _matches(self, dish: dict[str, Any], state: dict[str, Any]) -> bool:
        diet = state["diet"]
        tags = set(dish["tags"])
        if diet in {"vegan", "vegetarian"} and diet not in tags:
            return False
        if diet == "pescatarian" and "meat" in tags:
            return False
        if state["max_price"] is not None and dish["price"] > state["max_price"]:
            return False
        category = state["category"] or "main"
        if dish["category"] != category:
            return False
        if dish["category"] not in {"drink", "dessert"} and state["spice"] == "mild" and "mild" not in tags:
            return False
        if dish["category"] not in {"drink", "dessert"} and state["spice"] == "spicy" and not tags.intersection({"hot", "medium-spice"}):
            return False
        if state["caffeine"] == "caffeine-free" and "caffeine-free" not in tags:
            return False
        kind = state["dish_kind"]
        name = dish["name"].lower()
        if kind == "pasta" and not any(word in name for word in ("pasta", "ravioli", "linguine", "ragu")):
            return False
        if kind == "coffee" and not any(word in name for word in ("coffee", "espresso")):
            return False
        if kind == "tea" and "tea" not in name:
            return False
        if kind == "water" and "water" not in name:
            return False
        ingredients = set(dish.get("ingredients", [])) | set(dish["allergens"])
        if set(state["excluded_ingredients"]) & ingredients:
            return False
        return True

    def _recommend(self, text: str, state: dict[str, Any], allergy: bool = False) -> dict[str, Any]:
        candidates = [dish for dish in self.menu if self._matches(dish, state)]
        cheapest = bool(re.search(r"\b(?:cheapest|least expensive|lowest price|cheaper)\b", text))
        alternatives = bool(re.search(r"\b(?:another|alternatives?|something else|different ones?)\b", text))
        previous = state["recommended_ids"]
        if alternatives and previous:
            candidates = [item for item in candidates if item["id"] not in previous]
        elif cheapest and previous:
            previous_matches = [item for item in candidates if item["id"] in previous]
            if previous_matches:
                candidates = previous_matches
        candidates.sort(key=lambda item: (item["price"], item["name"]))
        choices = candidates[:1 if cheapest else 3]
        descriptors = [state["diet"]] if state["diet"] else []
        if state["max_price"] is not None:
            descriptors.append(f"at or below £{state['max_price']:.2f}")
        if state["spice"]:
            descriptors.append(state["spice"])
        if state["dish_kind"]:
            descriptors.append(state["dish_kind"])
        if state["caffeine"]:
            descriptors.append(state["caffeine"])
        if state["excluded_ingredients"]:
            descriptors.append("excluding listed " + ", ".join(state["excluded_ingredients"]))
        scope = "; ".join(descriptors) or "your preferences"
        if not choices:
            # Do not relax dietary restrictions silently when no dish matches.
            reply = f"I could not find a {state['category'] or 'main'} matching {scope} in this menu. Would you like to change the budget or choose another course?"
            if alternatives:
                reply = "There are no additional matches beyond the dishes already suggested. " + reply
            if allergy or state["excluded_ingredients"]:
                reply += "\n\n" + ALLERGY_NOTE
            state["recommended_ids"] = []
            state["selected_dish_id"] = None
            return self._result(reply, "recommendation", state, sources=[{"id": "menu", "title": "Seasonal menu"}],
                                prompts=["Remove the budget", "Show me desserts", "Forget my preferences"])
        lines = [f"Here {'is the lowest-priced match' if cheapest else 'are a few options'} for {scope}:"]
        for dish in choices:
            lines.append(f"• {dish['name']} — £{dish['price']:.2f}. {dish['description']}")
        lines.append("Would you like details about one of these, or should I adjust the suggestions?")
        if allergy or state["excluded_ingredients"]:
            lines.append(ALLERGY_NOTE)
        state["recommended_ids"] = [dish["id"] for dish in choices]
        state["selected_dish_id"] = choices[0]["id"] if len(choices) == 1 else None
        return self._result("\n\n".join(lines), "recommendation", state,
                            sources=[self._source(item) for item in choices],
                            prompts=["Tell me about the first one", "Which is cheapest?", "Show me desserts"])

    def _dish_details(self, dish: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        state["selected_dish_id"] = dish["id"]
        allergens = ", ".join(dish["allergens"]) or "none declared in this menu"
        reply = (f"{dish['name']} is £{dish['price']:.2f}. {dish['description']}\n\n"
                 f"Listed ingredients: {', '.join(dish['ingredients'])}.\n"
                 f"Declared allergens: {allergens}.\n\n{ALLERGY_NOTE}")
        return self._result(reply, "dish_details", state, sources=[self._source(dish)],
                            prompts=["Suggest something else", "Show me desserts", "How can I book a table?"])

    def reply(self, message: str, state: dict[str, Any] | None = None, mode: str = "enhanced") -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty text")
        if len(message) > 1200:
            raise ValueError("message must be at most 1200 characters")
        if mode not in {"enhanced", "baseline"}:
            raise ValueError("mode must be 'baseline' or 'enhanced'")
        state = {} if state is None else state
        self._ensure_state(state)
        text = normalise(message)

        if (re.fullmatch(r"/?(?:reset|restart)(?: (?:the |my )?conversation)?[!. ]*", text)
                or re.search(r"\b(?:forget my preferences|reset (?:my )?preferences|start over|start again|clear (?:my )?preferences)\b", text)):
            state.clear()
            state.update(self.new_state())
            return self._result("Your preferences have been cleared. What would you like to know about Olive Table?", "reset", state)
        if re.fullmatch(r"(?:hi|hello|hey|good (?:morning|afternoon|evening))(?: there)?[!. ]*", text):
            return self._result("Hello, welcome to Olive Table. I can help with the menu, dietary preferences or a question about your visit. What sounds good today?", "greeting", state)
        if re.fullmatch(r"(?:thanks|thank you|thank you very much|cheers|great thanks)[!. ]*", text):
            return self._result("You're very welcome. Let me know if you would like another suggestion or help planning your visit.", "thanks", state)
        if re.fullmatch(r"(?:bye|goodbye|see you|see you later)[!. ]*", text):
            return self._result("Goodbye, and thank you for visiting Olive Table. I hope you enjoy your meal!", "goodbye", state)

        # Safety-sensitive policy queries should receive the full policy rather
        # than a dish filter that could imply a guarantee.
        if re.search(r"\b(?:guarantee|cross[- ]?contamin|allergen (?:information|list)|allergy policy|handle (?:food )?allergies)\b", text):
            faq = self.faq_by_intent["allergens"]
            return self._result(faq["answer"], "allergens", state, sources=[self._source(faq)])

        changed, allergy = self._extract_preferences(text, state)
        dish = self._dish_mentioned(text, state)
        details_query = bool(re.search(r"\b(?:tell me|more|details|ingredients?|contains?|allergens?|price|cost|how much|what(?:'s| is) in|first|second|third)\b", text))
        if dish and (details_query or not changed):
            return self._dish_details(dish, state)

        recommendation_query = bool(re.search(
            r"\b(?:recommend\w*|suggest\w*|hungry|eat|vegan|vegetarian|pescatarian|plant[- ]based|cheapest|cheaper|affordable|something else|alternatives?|another|desserts?|starters?|appeti[sz]ers?|drinks?|beverages?|sides?)\b", text))
        continuation = bool(state["recommended_ids"] and re.search(r"\b(?:under|below|less than|budget|mild|spicy|cheaper|different|instead|that sounds|yes|any price|remove)\b", text))
        if changed or recommendation_query or continuation:
            return self._recommend(text, state, allergy=allergy)
        if re.search(r"\b(?:it|that one|the first one|the second one)\b", text) and state["recommended_ids"]:
            return self._result("Which dish do you mean? You can use its name or say 'the first one', 'the second one' or 'the third one'.", "clarification", state)

        # The enhanced application routes FAQ questions with a supervised,
        # locally trained classifier when supplied. Raw TF-IDF remains exposed
        # through retrieve() for an honest, reproducible ablation comparison.
        match = self.retrieve(text, mode)
        method = "Word-unigram TF-IDF retrieval" if mode == "baseline" else "Hybrid retrieval (trained model unavailable)"
        if mode == "enhanced" and self.classifier is not None:
            prediction = self.classifier.predict(text)
            method = prediction.get("method", "Locally trained intent classifier and FAQ grounding")
            intent = prediction.get("intent", "fallback")
            accepted = prediction.get("accepted", intent != "fallback")
            match = {"intent": intent if accepted else "fallback", "confidence": prediction.get("score", 0.0)}
            if accepted and intent == "recommendation":
                return self._recommend(text, state, allergy=allergy)
            social = {
                "greeting": "Hello, welcome to Olive Table. I can help with dishes or a question about your visit. What would you like to know?",
                "thanks": "You're very welcome. Is there anything else I can help you with?",
                "goodbye": "Goodbye, and thank you for visiting Olive Table!",
            }
            if accepted and intent in social:
                return self._result(social[intent], intent, state, match["confidence"], method=method)
        if match["intent"] in self.faq_by_intent:
            faq = self.faq_by_intent[match["intent"]]
            return self._result(faq["answer"], faq["intent"], state, match["confidence"], [self._source(faq)], method=method)
        return self._result(
            "I couldn't find a reliable answer to that in our restaurant information. I can help with dishes, opening hours, dietary preferences or table enquiries. Could you rephrase your restaurant question?",
            "fallback", state, match["confidence"],
            prompts=["Show me the menu", "When are you open?", "I am vegan and my budget is 15 pounds"],
            method=method,
        )
