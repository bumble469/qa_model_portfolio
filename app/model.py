import json
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "qa_data.json"
MODEL_PATH = BASE_DIR / "models" / "qa_model.pkl"

vectorizer: TfidfVectorizer | None = None
question_vectors = None
answers = []
questions = []

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

    questions = [item["question"] for item in qa_data]
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


def get_answer(user_question: str, treshold: float = 0.2, top_k: int = 2):
    if vectorizer is None or question_vectors is None:
        raise RuntimeError("Model not loaded")

    subject = detect_subject(user_question)
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    ranked_indices = similarities.argsort()[::-1]

    selected_answers = []
    selected_scores = []

    for i in ranked_indices:
        q_text = questions[i].lower()

        if subject == "alisher" and "alisher" not in q_text:
            continue
        if subject == "assistant" and not (
            "you" in q_text or "yourself" in q_text
        ):
            continue

        if similarities[i] >= treshold:
            selected_answers.append(answers[i])
            selected_scores.append(similarities[i])

        if len(selected_answers) >= top_k:
            break

    if not selected_answers:
        return (
            "I can answer questions about Alisher’s education, projects, skills, and experience.",
            0.0
        )

    return selected_answers[0], float(max(selected_scores))