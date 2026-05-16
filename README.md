# NLP Restaurant Chatbot 🤖🍽️

An intelligent, task-oriented NLP chatbot framework designed to handle restaurant reservations, answer queries, and manage customer interactions. This project combines traditional NLP pipelines (Intent Classification, Named Entity Recognition) with modern Large Language Model (LLM) fine-tuning capabilities using Llama 3.1.

## 🚀 Features

- **Custom Intent Classification**: Categorizes user inputs like booking a table or cancelling a reservation.
- **Named Entity Recognition (NER)**: Extracts specific entities like dietary requirements (e.g., vegetarian, gluten-free), time, and party size.
- **Pipeline Architecture**: Modular `src/` directory containing scripts for preprocessing, sentiment analysis, and dialogue management.
- **LLM Fine-tuning Ready**: Includes Jupyter Notebooks configured for 4-bit QLoRA fine-tuning of `Meta-Llama-3.1-8B-Instruct` to create a more fluid, generative conversational experience.
- **Web Interface**: A starting template for a web-based chat UI using HTML, CSS, and vanilla JavaScript.

## 📂 Project Structure

```text
nlp-chatbot/
│
├── data/                   # Custom datasets
│   ├── entities.json       # NER training data (e.g., dietary constraints)
│   ├── intents.json        # Intent classification data
│   └── responses.json      # Hardcoded bot responses for the rules-based pipeline
│
├── models/                 
│   └── bert_intent/        # Directory for saving fine-tuned BERT/LLM weights
│
├── notebooks/              # Jupyter Notebooks for EDA and Model Training
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_llama3_qlora.ipynb   # Code for fine-tuning Llama 3.1 using QLoRA
│
├── src/                    # Core NLP Pipeline code
│   ├── dialogue.py         # Manages multi-turn conversation flow
│   ├── intent_classifier.py# Predicts user intent
│   ├── ner.py              # Extracts entities
│   ├── preprocess.py       # Text cleaning and tokenization
│   ├── response.py         # Formats bot replies
│   └── sentiment.py        # Analyzes user sentiment
│
├── static/                 # Frontend Web UI
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── tests/                  # Unit tests for the pipeline
│
├── main.py                 # Main application entry point (CLI Chat loop)
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup script
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd nlp-chatbot
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Authenticate with Hugging Face:**
   If you plan to use the Llama 3.1 notebook (`02_llama3_qlora.ipynb`), you must log in to Hugging Face via the CLI using your access token:
   ```bash
   hf auth login
   ```

## 💻 Usage

### Running the Standard Pipeline (CLI)
To run the standard intent-based chatbot in your terminal:
```bash
python main.py
```
Type your query (e.g., "I want to book a table for 2 tonight") and interact with the bot. Type `exit` to quit.

### Fine-Tuning Llama 3.1
To experiment with generative AI:
1. Open `notebooks/02_llama3_qlora.ipynb`.
2. Ensure you have accepted the Meta Llama 3.1 license on Hugging Face.
3. Run the notebook to load the model in 4-bit precision and prepare it for LoRA training on consumer GPUs.

### Web Interface
To view the front-end chat interface, simply open `static/index.html` in your web browser. (Note: API endpoints connecting the frontend to the backend need to be implemented via a framework like FastAPI/Flask).
