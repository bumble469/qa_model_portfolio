import json
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "qa_data.json"
MODEL_PATH = BASE_DIR / "models" / "qa_model.pkl"

vectorizer: TfidfVectorizer | None = None
question_vectors = None
answers = []
questions = []

lemmatizer = WordNetLemmatizer()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def load_or_train_model():
    global vectorizer, question_vectors, answers, questions

    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as m:
            vectorizer, question_vectors, questions, answers = pickle.load(m)
        return

    train_model()

def train_model():
    global vectorizer, question_vectors, answers, questions

    with open(DATA_PATH, "r", encoding="utf-8") as d:
        qa_data = json.load(d)

    raw_questions = [item["question"] for item in qa_data]
    questions = [normalize_text(q) for q in raw_questions]
    answers = [item["answer"] for item in qa_data]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    question_vectors = vectorizer.fit_transform(questions)

    with open(MODEL_PATH, "wb") as m:
        pickle.dump((vectorizer, question_vectors, questions, answers), m)


def detect_subject(q: str):
    q = q.lower()
    if "alisher" in q:
        return "alisher"
    if "you" in q or "yourself" in q:
        return "assistant"
    return "neutral"


def get_answer(user_question: str, threshold: float = 0.25):
    if vectorizer is None or question_vectors is None:
        raise RuntimeError("Model not loaded")

    subject = detect_subject(user_question)
    normalized_question = normalize_text(user_question)
    user_vector = vectorizer.transform([normalized_question])

    similarities = cosine_similarity(user_vector, question_vectors)[0]
    ranked_indices = similarities.argsort()[::-1]

    for i in ranked_indices:
        q_text = questions[i].lower()

        if subject == "alisher" and "alisher" not in q_text:
            continue
        if subject == "assistant" and not ("you" in q_text or "yourself" in q_text):
            continue

        if similarities[i] < threshold:
            continue

        return answers[i].strip(), float(similarities[i])

    return (
        "I can answer questions about Alisher’s education, projects, skills, and experience.",
        0.0
    )

