"""
Task 2: Chatbot for FAQs
------------------------
A retrieval-based FAQ chatbot. It stores a set of (question, answer) pairs,
preprocesses text with NLTK (tokenize, lowercase, remove stopwords,
lemmatize), and matches the user's question to the closest FAQ using
TF-IDF + cosine similarity.

Run the Streamlit chat UI with:
    streamlit run app.py

Or run as a plain command-line chatbot with:
    python app.py --cli
"""

import sys
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# One-time NLTK data download (safe to call every run; it skips if present)
# ---------------------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# 1. FAQ Data — replace/extend this with your own product's FAQs
# ---------------------------------------------------------------------------
FAQS = [
    {"question": "What is your return policy?",
     "answer": "You can return any item within 30 days of purchase for a full refund, as long as it is unused and in its original packaging."},
    {"question": "How do I track my order?",
     "answer": "Once your order ships, you'll receive an email with a tracking number and a link to track it in real time."},
    {"question": "Do you offer international shipping?",
     "answer": "Yes, we ship to over 50 countries. Shipping costs and delivery times vary by destination."},
    {"question": "How can I contact customer support?",
     "answer": "You can reach our support team via live chat on our website, or email us at support@example.com."},
    {"question": "What payment methods do you accept?",
     "answer": "We accept all major credit cards, PayPal, and Apple Pay."},
    {"question": "How do I reset my password?",
     "answer": "Click 'Forgot Password' on the login page and follow the instructions sent to your registered email."},
    {"question": "Can I change or cancel my order after placing it?",
     "answer": "Orders can be changed or cancelled within 1 hour of purchase. After that, please contact support."},
    {"question": "Do you offer a warranty on your products?",
     "answer": "Most products come with a 1-year manufacturer's warranty covering defects in materials and workmanship."},
]

SIMILARITY_THRESHOLD = 0.2  # below this, the bot admits it doesn't know


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Lowercase, tokenize, strip punctuation/stopwords, and lemmatize."""
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in string.punctuation and tok not in STOPWORDS and tok.isalpha()
    ]
    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# 3. Matching engine (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------------
class FAQBot:
    def __init__(self, faqs):
        self.faqs = faqs
        self.questions_raw = [f["question"] for f in faqs]
        self.questions_clean = [preprocess(q) for q in self.questions_raw]
        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self.vectorizer.fit_transform(self.questions_clean)

    def get_answer(self, user_question: str):
        cleaned = preprocess(user_question)
        if not cleaned.strip():
            return "Could you rephrase your question?", 0.0, None

        user_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(user_vec, self.faq_vectors).flatten()
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score < SIMILARITY_THRESHOLD:
            return (
                "Sorry, I don't have an answer for that yet. "
                "Please contact support@example.com for further help.",
                best_score,
                None,
            )
        return self.faqs[best_idx]["answer"], best_score, self.faqs[best_idx]["question"]


# ---------------------------------------------------------------------------
# 4a. Command-line interface
# ---------------------------------------------------------------------------
def run_cli():
    bot = FAQBot(FAQS)
    print("FAQ Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        answer, score, matched_q = bot.get_answer(user_input)
        print(f"Bot: {answer}")


# ---------------------------------------------------------------------------
# 4b. Streamlit chat UI
# ---------------------------------------------------------------------------
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
    st.title("🤖 FAQ Chatbot")
    st.caption("Ask a question and I'll try to match it to the closest FAQ.")

    if "bot" not in st.session_state:
        st.session_state.bot = FAQBot(FAQS)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me anything about our product or service."}
        ]

    with st.sidebar:
        st.subheader("Available FAQs")
        for f in FAQS:
            st.markdown(f"- {f['question']}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        answer, score, matched_q = st.session_state.bot.get_answer(prompt)
        with st.chat_message("assistant"):
            st.write(answer)
            if matched_q:
                st.caption(f"Matched FAQ: \"{matched_q}\" (similarity: {score:.2f})")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_streamlit()
