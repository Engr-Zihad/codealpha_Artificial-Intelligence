# Task 2: Chatbot for FAQs

A retrieval-based FAQ chatbot that matches a user's question to the most
similar entry in a predefined FAQ list using **TF-IDF + cosine similarity**,
after cleaning the text with **NLTK** (tokenization, stopword removal,
lemmatization).

## Features
- A list of sample FAQs (`question` + `answer` pairs) in `app.py` — replace
  these with your own product/topic FAQs
- NLP preprocessing: lowercasing, tokenization, punctuation/stopword removal,
  lemmatization
- TF-IDF vectorization of all FAQ questions
- Cosine similarity to find the best-matching FAQ for a user's question
- A similarity threshold so the bot admits when it doesn't know the answer
- Two interfaces:
  - **Streamlit chat UI** (default)
  - **Command-line chatbot** (`python app.py --cli`)

## Setup

```bash
cd Task_2_FAQ_Chatbot
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run automatically downloads the small NLTK data packages it needs
(`punkt`, `stopwords`, `wordnet`).

## Run

**Chat UI:**
```bash
streamlit run app.py
```

**Command line:**
```bash
python app.py --cli
```

## Customizing
- Edit the `FAQS` list at the top of `app.py` to add your own questions and
  answers.
- Adjust `SIMILARITY_THRESHOLD` (default `0.2`) to make matching stricter or
  more lenient.
- For more advanced intent matching, you could swap TF-IDF for sentence
  embeddings (e.g. `sentence-transformers`) and use cosine similarity on
  the embedding vectors instead.
