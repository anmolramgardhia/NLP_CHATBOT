"""
preprocess.py — Text cleaning and normalisation module
NLP Chatbot Project — Restaurant Booking Domain
"""

import re
import string

CONTRACTIONS = {
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "that's": "that is", "there's": "there is", "it's": "it is",
    "what's": "what is", "where's": "where is", "when's": "when is",
    "how's": "how is", "let's": "let us", "we're": "we are",
    "we've": "we have", "we'll": "we will", "they're": "they are",
    "they've": "they have", "they'll": "they will", "you're": "you are",
    "you've": "you have", "you'll": "you will", "you'd": "you would",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS) + r')\b',
        re.IGNORECASE
    )
    return pattern.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)


def clean_text(text: str) -> str:
    """
    Clean and normalise raw user input for NLP processing.

    Steps:
        1. Strip and lowercase
        2. Expand contractions
        3. Collapse excess whitespace
        Preserves punctuation for spaCy NER boundary detection.

    Args:
        text: Raw user message.

    Returns:
        Cleaned normalised string, or "" for invalid input.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.strip().lower()
    text = expand_contractions(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenise(text: str) -> list:
    """Split cleaned text into word tokens."""
    return text.split() if text else []


def preprocess(text: str) -> dict:
    """
    Full preprocessing pipeline.

    Args:
        text: Raw user input.

    Returns:
        Dict with keys: original, cleaned, tokens.
    """
    cleaned = clean_text(text)
    return {
        "original": text,
        "cleaned":  cleaned,
        "tokens":   tokenise(cleaned),
    }


if __name__ == "__main__":
    samples = [
        "I'd like to book a table for 2 at 7pm tomorrow!",
        "Can't you fit us in on Saturday?",
        "What's your address?",
    ]
    for s in samples:
        r = preprocess(s)
        print(f"Original : {r['original']}")
        print(f"Cleaned  : {r['cleaned']}")
        print(f"Tokens   : {r['tokens']}\n")
