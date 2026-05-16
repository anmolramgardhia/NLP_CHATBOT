"""
dialogue.py — Dialogue state tracker and slot-filling module
NLP Chatbot Project — Restaurant Booking Domain

The DialogueStateTracker maintains per-session conversation state,
fills required slots for each intent, and generates clarifying questions
when slots are missing before passing control to response generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


REQUIRED_SLOTS: dict[str, list[str]] = {
    "book_table":         ["party_size", "date", "time"],
    "cancel_booking":     ["person_name"],
    "modify_booking":     ["person_name"],
    "check_availability": ["date"],
    "greet":              [],
    "goodbye":            [],
    "ask_menu":           [],
    "ask_hours":          [],
    "ask_location":       [],
    "out_of_scope":       [],
}

CLARIFYING_QUESTIONS: dict[str, str] = {
    "party_size":  "How many people will be dining?",
    "date":        "What date were you thinking?",
    "time":        "What time would you like the table?",
    "person_name": "Could you give me the name the booking is under?",
}


@dataclass
class DialogueState:
    """Represents the full state of a single user session."""
    session_id:      str
    current_intent:  Optional[str] = None
    slots:           dict          = field(default_factory=dict)
    history:         list          = field(default_factory=list)
    turn_count:      int           = 0
    is_complete:     bool          = False

    def update_slots(self, new_slots: dict):
        """Merge newly extracted entities into the current slot dict."""
        for key, value in new_slots.items():
            if key in ("raw", "dietary"):
                continue
            if value is not None and self.slots.get(key) is None:
                self.slots[key] = value
        if "dietary" in new_slots and new_slots["dietary"]:
            self.slots.setdefault("dietary", [])
            for d in new_slots["dietary"]:
                if d not in self.slots["dietary"]:
                    self.slots["dietary"].append(d)

    def missing_slots(self) -> list[str]:
        """Return required slots not yet filled for the current intent."""
        if self.current_intent not in REQUIRED_SLOTS:
            return []
        return [s for s in REQUIRED_SLOTS[self.current_intent] if not self.slots.get(s)]

    def add_turn(self, role: str, text: str):
        """Append a turn to the conversation history."""
        self.history.append({"role": role, "text": text})
        if role == "user":
            self.turn_count += 1

    def reset(self):
        """Reset slot state but preserve conversation history."""
        self.slots          = {}
        self.current_intent = None
        self.is_complete    = False


class DialogueStateTracker:
    """
    Manages dialogue state across multiple sessions.
    In production this is backed by Redis via memory.py.
    """

    def __init__(self):
        self._sessions: dict[str, DialogueState] = {}

    def get_or_create(self, session_id: str) -> DialogueState:
        if session_id not in self._sessions:
            self._sessions[session_id] = DialogueState(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].reset()

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)

    def process_turn(
        self,
        session_id: str,
        user_text:  str,
        intent:     str,
        entities:   dict,
        confidence: float,
    ) -> dict:
        """
        Process one dialogue turn and decide the next action.

        Returns:
            Dict with action ('respond'|'clarify'), missing_slot,
            clarifying_question, and state.
        """
        state = self.get_or_create(session_id)
        state.add_turn("user", user_text)

        # Reset slots if intent changes (except out_of_scope)
        if intent != state.current_intent and intent != "out_of_scope":
            state.slots         = {}
            state.current_intent = intent

        state.update_slots(entities)
        missing = state.missing_slots()

        if missing:
            first_missing = missing[0]
            question = CLARIFYING_QUESTIONS.get(
                first_missing,
                f"Could you provide your {first_missing.replace('_', ' ')}?"
            )
            return {
                "action":               "clarify",
                "missing_slot":         first_missing,
                "clarifying_question":  question,
                "state":                state,
            }

        state.is_complete = True
        return {
            "action":               "respond",
            "missing_slot":         None,
            "clarifying_question":  None,
            "state":                state,
        }


if __name__ == "__main__":
    tracker = DialogueStateTracker()
    turns = [
        ("u1", "I would like to book a table", "book_table", {}),
        ("u1", "For two people", "book_table", {"party_size": "2"}),
        ("u1", "Tomorrow evening", "book_table", {"date": "tomorrow"}),
        ("u1", "At 7pm please", "book_table", {"time": "7pm"}),
    ]
    for sid, text, intent, entities in turns:
        r = tracker.process_turn(sid, text, intent, entities, 0.95)
        print(f"User: {text}")
        if r["action"] == "clarify":
            print(f"  Bot : {r['clarifying_question']}")
        else:
            print(f"  Slots: {r['state'].slots}")
        print()
