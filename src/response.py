"""
response.py — Response generation module
NLP Chatbot Project — Restaurant Booking Domain

Template-based responses for known intents.
LLM fallback for out_of_scope or low-confidence predictions.
"""

import json
import random
from pathlib import Path

RESPONSES_PATH = Path(__file__).parent.parent / "data" / "responses.json"


class ResponseGenerator:
    """
    Generates responses using templates (known intents)
    or an LLM API fallback (out_of_scope).
    """

    def __init__(self, responses_path: str = None):
        path = Path(responses_path) if responses_path else RESPONSES_PATH
        with open(path, encoding="utf-8") as f:
            self._templates = json.load(f)["responses"]

    def generate(self, intent: str, state, sentiment_label: str = "neutral") -> str:
        """
        Generate a response for a completed dialogue turn.

        Args:
            intent:          Classified intent label.
            state:           DialogueState with filled slots.
            sentiment_label: User sentiment string.

        Returns:
            Response string.
        """
        prefix = ""
        if sentiment_label == "frustrated":
            prefix = "I'm really sorry to hear that. Let me help you right away. "

        slots = state.slots

        if intent == "greet":
            return prefix + random.choice(self._templates["greet"])

        if intent == "goodbye":
            return prefix + random.choice(self._templates["goodbye"])

        if intent in ("ask_menu", "ask_hours", "ask_location"):
            return prefix + random.choice(self._templates[intent])

        if intent == "book_table":
            ref = str(random.randint(10000, 99999))
            return prefix + self._templates["book_table"]["confirm"].format(
                party_size=slots.get("party_size", "your party"),
                date=slots.get("date", "the requested date"),
                time=slots.get("time", "the requested time"),
                ref=ref,
            )

        if intent == "cancel_booking":
            return prefix + self._templates["cancel_booking"]["confirm"].format(
                date=slots.get("date", "your booking date"),
                time=slots.get("time", "your booking time"),
            )

        if intent == "modify_booking":
            return prefix + self._templates["modify_booking"]["confirm"].format(
                party_size=slots.get("party_size", "your party"),
                date=slots.get("date", "the requested date"),
                time=slots.get("time", "the requested time"),
            )

        if intent == "check_availability":
            return prefix + self._templates["check_availability"]["available"].format(
                date=slots.get("date", "that date"),
                time=slots.get("time", "your preferred time"),
                max_party="10",
            )

        return prefix + random.choice(self._templates["out_of_scope"])

    def clarify(self, question: str, sentiment_label: str = "neutral") -> str:
        """Return a clarifying question, softened for frustrated users."""
        if sentiment_label == "frustrated":
            return f"I want to make sure I get this right for you — {question.lower()}"
        return question

    def llm_fallback(self, conversation_history: list, user_message: str) -> str:
        """
        Use the Anthropic API to handle out-of-scope inputs.
        Set ANTHROPIC_API_KEY environment variable to enable.
        """
        try:
            import anthropic, os
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            system = (
                "You are a helpful assistant for The Grand Table restaurant. "
                "Help customers with bookings, menu questions, and general enquiries. "
                "Keep responses concise and friendly. "
                "If asked something unrelated to the restaurant, politely redirect."
            )
            messages = [
                {"role": m["role"] if m["role"] in ("user", "assistant") else "user",
                 "content": m["text"]}
                for m in conversation_history[-6:]
            ]
            messages.append({"role": "user", "content": user_message})
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=256,
                system=system, messages=messages,
            )
            return resp.content[0].text
        except Exception as e:
            print(f"LLM fallback error: {e}")
            return random.choice(self._templates["out_of_scope"])


if __name__ == "__main__":
    from dialogue import DialogueState
    gen   = ResponseGenerator()
    state = DialogueState(session_id="test")
    state.current_intent = "book_table"
    state.slots = {"party_size": "2", "date": "tomorrow", "time": "7pm"}
    print(gen.generate("book_table", state))
    print(gen.generate("greet", state))
    print(gen.generate("out_of_scope", state))
