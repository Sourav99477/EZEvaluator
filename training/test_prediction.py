import numpy as np
import joblib
from sklearn.preprocessing import normalize

embedding_model = joblib.load("../model/embedding_model.pkl")
tfidf = joblib.load("../model/tfidf_vectorizer.pkl")
stage1_model = joblib.load("../model/stage1_model.pkl")
stage2_model = joblib.load("../model/stage2_model.pkl")

questions = np.array([
    "What is photosynthesis? Explain the process briefly.",
    "What is the difference between evaporation and boiling?",
    "What is friction? Give one example from everyday life."
])

references = np.array([
    "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to make food and release oxygen.",
    "Evaporation is the slow conversion of a liquid into vapor from its surface, while boiling is rapid vaporization throughout the liquid at its boiling point.",
    "Friction is a force that opposes motion between two surfaces. An example is friction between bicycle brakes and the wheel."
])

students = np.array([
    "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to make food and release oxygen.",
    "Evaporation occurs slowly at the surface of a liquid, while boiling occurs rapidly throughout the liquid at its boiling point.",
    "Friction is a force that opposes motion between surfaces, such as when brakes stop a bicycle."
])

def build_embedding_features(q, ref, stu):
    q_ref = np.abs(q - ref)
    q_stu = np.abs(q - stu)
    ref_stu = np.abs(ref - stu)

    q_ref_product = q * ref
    q_stu_product = q * stu
    ref_stu_product = ref * stu

    q_ref_cosine = np.sum(q * ref, axis=1, keepdims=True)
    q_stu_cosine = np.sum(q * stu, axis=1, keepdims=True)
    ref_stu_cosine = np.sum(ref * stu, axis=1, keepdims=True)

    return np.hstack([
        q_ref,
        q_stu,
        ref_stu,
        q_ref_product,
        q_stu_product,
        ref_stu_product,
        q_ref_cosine,
        q_stu_cosine,
        ref_stu_cosine
    ])

def cosine_features(q, ref, stu):
    q_ref = np.sum(q.multiply(ref), axis=1).A
    q_stu = np.sum(q.multiply(stu), axis=1).A
    ref_stu = np.sum(ref.multiply(stu), axis=1).A

    return np.column_stack([
        q_ref,
        q_stu,
        ref_stu
    ])

def text_features(ref, stu):
    ref_len = np.array([len(x.split()) for x in ref])
    stu_len = np.array([len(x.split()) for x in stu])

    length_ratio = (stu_len + 1) / (ref_len + 1)

    overlap = []

    for r, s in zip(ref, stu):
        r_words = set(r.lower().split())
        s_words = set(s.lower().split())

        if not r_words:
            overlap.append(0)
        else:
            overlap.append(len(r_words & s_words) / len(r_words))

    return np.column_stack([
        ref_len,
        stu_len,
        length_ratio,
        overlap
    ])

q_emb = embedding_model.encode(
    questions.tolist(),
    normalize_embeddings=True
)

ref_emb = embedding_model.encode(
    references.tolist(),
    normalize_embeddings=True
)

stu_emb = embedding_model.encode(
    students.tolist(),
    normalize_embeddings=True
)

embedding_features = build_embedding_features(
    q_emb,
    ref_emb,
    stu_emb
)

q_tfidf = normalize(tfidf.transform(questions))
ref_tfidf = normalize(tfidf.transform(references))
stu_tfidf = normalize(tfidf.transform(students))

tfidf_features = cosine_features(
    q_tfidf,
    ref_tfidf,
    stu_tfidf
)

text_features_result = text_features(
    references,
    students
)

X = np.hstack([
    embedding_features,
    tfidf_features,
    text_features_result
])

stage1_predictions = stage1_model.predict(X)

final_predictions = []

for i, prediction in enumerate(stage1_predictions):
    if prediction == 0:
        final_predictions.append(0)
    else:
        stage2_prediction = stage2_model.predict(
            X[i:i + 1]
        )[0]

        if stage2_prediction == 0:
            final_predictions.append(1)
        else:
            final_predictions.append(2)

labels = {
    0: "Likely Correct",
    1: "Needs Teacher Review",
    2: "Likely Incorrect"
}

print("\nEZEvaluator Results\n")

for i, prediction in enumerate(final_predictions):
    print(f"Question {i + 1}: {labels[prediction]}")