"""
ner.py — Named Entity Recognition module
NLP Chatbot Project — Restaurant Booking Domain

Uses spaCy en_core_web_trf for standard entities (DATE, TIME, CARDINAL, PERSON).
Extended with EntityRuler patterns for domain-specific DIETARY entities.

Setup: python -m spacy download en_core_web_trf
       (falls back to en_core_web_sm if trf unavailable)
"""

from __future__ import annotations
from typing import Optional
import spacy

SPACY_MODEL = "en_core_web_trf"

DIETARY_PATTERNS = [
    {"label": "DIETARY", "pattern": "vegetarian"},
    {"label": "DIETARY", "pattern": "vegan"},
    {"label": "DIETARY", "pattern": "plant-based"},
    {"label": "DIETARY", "pattern": "gluten-free"},
    {"label": "DIETARY", "pattern": "gluten free"},
    {"label": "DIETARY", "pattern": "dairy-free"},
    {"label": "DIETARY", "pattern": "dairy free"},
    {"label": "DIETARY", "pattern": "nut-free"},
    {"label": "DIETARY", "pattern": "halal"},
    {"label": "DIETARY", "pattern": "kosher"},
    {"label": "DIETARY", "pattern": "pescatarian"},
]


class NERExtractor:
    """
    Named entity extractor using spaCy + custom EntityRuler.
    Extracts: DATE, TIME, CARDINAL (party size), PERSON, DIETARY.
    """

    def __init__(self, model: str = SPACY_MODEL):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"  spaCy model '{model}' not found. Falling back to en_core_web_sm.")
            print("  Install full model: python -m spacy download en_core_web_trf")
            self.nlp = spacy.load("en_core_web_sm")

        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(DIETARY_PATTERNS)

    def extract(self, text: str) -> dict:
        """
        Extract named entities from cleaned text.

        Args:
            text: Cleaned user message.

        Returns:
            Dict with keys: date, time, party_size, person_name, dietary, raw.
        """
        doc = self.nlp(text)
        entities = {
            "date":        None,
            "time":        None,
            "party_size":  None,
            "person_name": None,
            "dietary":     [],
            "raw":         [],
        }

        for ent in doc.ents:
            entities["raw"].append({
                "text": ent.text, "label": ent.label_,
                "start": ent.start_char, "end": ent.end_char,
            })
            if ent.label_ == "DATE" and not entities["date"]:
                entities["date"] = ent.text
            elif ent.label_ == "TIME" and not entities["time"]:
                entities["time"] = ent.text
            elif ent.label_ == "CARDINAL" and not entities["party_size"]:
                try:
                    n = int(ent.text.split()[0])
                    if 1 <= n <= 30:
                        entities["party_size"] = str(n)
                except (ValueError, IndexError):
                    pass
            elif ent.label_ == "PERSON" and not entities["person_name"]:
                entities["person_name"] = ent.text
            elif ent.label_ == "DIETARY":
                if ent.text.lower() not in entities["dietary"]:
                    entities["dietary"].append(ent.text.lower())

        return entities

    def extract_party_size_from_text(self, text: str) -> Optional[str]:
        """Word-to-number fallback for party size detection."""
        word_map = {
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10", "a couple": "2", "just me": "1",
        }
        text_lower = text.lower()
        for word, num in word_map.items():
            if word in text_lower:
                return num
        return None


if __name__ == "__main__":
    extractor = NERExtractor()
    samples = [
        "Book a table for 2 at 7pm tomorrow",
        "Reserve a spot for four people on Friday evening",
        "I need a vegan menu option for my booking under John Smith",
    ]
    for text in samples:
        r = extractor.extract(text)
        print(f"Text  : {text}")
        print(f"  date={r['date']}  time={r['time']}  "
              f"party={r['party_size']}  person={r['person_name']}  dietary={r['dietary']}\n")
