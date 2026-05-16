import argparse
import sys
import logging

from src.dialogue import DialogueManager
from src.intent_classifier import IntentClassifier
from src.ner import NamedEntityRecognizer
from src.sentiment import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting up the NLP Chatbot...")
    
    # Initialize models and managers (assuming structure)
    try:
        dialogue_manager = DialogueManager()
        intent_classifier = IntentClassifier()
        ner = NamedEntityRecognizer()
        
        print("Chatbot initialized! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Chatbot: Goodbye!")
                break
                
            # Basic pipeline loop
            # intent = intent_classifier.predict(user_input)
            # entities = ner.extract(user_input)
            # response = dialogue_manager.get_response(intent, entities)
            
            # print(f"Chatbot: {response}")
            
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
