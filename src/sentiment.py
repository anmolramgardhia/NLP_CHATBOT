"""
sentiment.py — Sentiment analysis module
NLP Chatbot Project — Restaurant Booking Domain

Uses VADER for fast rule-based sentiment scoring. No training required.
Install: pip install vaderSentiment
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD   =  0.05
NEGATIVE_THRESHOLD   = -0.05
FRUSTRATED_THRESHOLD = -0.4    # strong negative → empathetic response


class SentimentDetector:
    """
    Detects user sentiment: positive / neutral / negative / frustrated.
    'frustrated' triggers a more empathetic response template.
    """

    def __init__(self):
        self.analyser = SentimentIntensityAnalyzer()

    def analyse(self, text: str) -> dict:
        """
        Analyse sentiment of a user message.

        Args:
            text: Cleaned user message.

        Returns:
            Dict with label, compound score, raw scores, is_frustrated flag.
        """
        scores   = self.analyser.polarity_scores(text)
        compound = scores["compound"]

        if compound >= POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= FRUSTRATED_THRESHOLD:
            label = "frustrated"
        elif compound <= NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label":        label,
            "compound":     round(compound, 4),
            "scores":       {k: round(v, 4) for k, v in scores.items()},
            "is_frustrated": label == "frustrated",
        }


if __name__ == "__main__":
    detector = SentimentDetector()
    samples  = [
        "I'd love to book a table!",
        "Book me a table",
        "This is ridiculous, I have been waiting for ages",
        "Thanks so much, you have been really helpful!",
    ]
    for text in samples:
        r = detector.analyse(text)
        print(f"Text      : {text}")
        print(f"Sentiment : {r['label']} (compound={r['compound']})\n")
