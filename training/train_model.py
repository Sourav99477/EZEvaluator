import pandas as pd
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize


df = pd.read_parquet("train-00001.parquet")

df["question"] = df["question"].fillna("").astype(str)
df["reference_answer"] = df["reference_answer"].fillna("").astype(str)
df["student_answer"] = df["student_answer"].fillna("").astype(str)

question = df["question"].to_numpy()
reference = df["reference_answer"].to_numpy()
student = df["student_answer"].to_numpy()

original_label = df["label"].astype(int).to_numpy()

stage1_label = np.where(original_label == 0, 0, 1)

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=stage1_label
)

q_train = question[train_idx]
q_test = question[test_idx]

ref_train = reference[train_idx]
ref_test = reference[test_idx]

stu_train = student[train_idx]
stu_test = student[test_idx]

y1_train = stage1_label[train_idx]
y1_test = stage1_label[test_idx]

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating question embeddings...")

q_train_emb = embedding_model.encode(
    q_train.tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

q_test_emb = embedding_model.encode(
    q_test.tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Creating reference embeddings...")

ref_train_emb = embedding_model.encode(
    ref_train.tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

ref_test_emb = embedding_model.encode(
    ref_test.tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Creating student embeddings...")

stu_train_emb = embedding_model.encode(
    stu_train.tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

stu_test_emb = embedding_model.encode(
    stu_test.tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)


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


train_embedding_features = build_embedding_features(
    q_train_emb,
    ref_train_emb,
    stu_train_emb
)

test_embedding_features = build_embedding_features(
    q_test_emb,
    ref_test_emb,
    stu_test_emb
)


print("Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=15000
)

tfidf.fit(
    np.concatenate([
        q_train,
        ref_train,
        stu_train
    ])
)

q_train_tfidf = normalize(tfidf.transform(q_train))
ref_train_tfidf = normalize(tfidf.transform(ref_train))
stu_train_tfidf = normalize(tfidf.transform(stu_train))

q_test_tfidf = normalize(tfidf.transform(q_test))
ref_test_tfidf = normalize(tfidf.transform(ref_test))
stu_test_tfidf = normalize(tfidf.transform(stu_test))


def cosine_features(q, ref, stu):
    q_ref = np.sum(
        q.multiply(ref),
        axis=1
    ).A

    q_stu = np.sum(
        q.multiply(stu),
        axis=1
    ).A

    ref_stu = np.sum(
        ref.multiply(stu),
        axis=1
    ).A

    return np.column_stack([
        q_ref,
        q_stu,
        ref_stu
    ])


train_tfidf_features = cosine_features(
    q_train_tfidf,
    ref_train_tfidf,
    stu_train_tfidf
)

test_tfidf_features = cosine_features(
    q_test_tfidf,
    ref_test_tfidf,
    stu_test_tfidf
)


def text_features(ref, stu):
    ref_len = np.array([
        len(x.split())
        for x in ref
    ])

    stu_len = np.array([
        len(x.split())
        for x in stu
    ])

    length_ratio = (
        stu_len + 1
    ) / (
        ref_len + 1
    )

    overlap = []

    for r, s in zip(ref, stu):
        r_words = set(r.lower().split())
        s_words = set(s.lower().split())

        if not r_words:
            overlap.append(0)
        else:
            overlap.append(
                len(r_words & s_words) / len(r_words)
            )

    return np.column_stack([
        ref_len,
        stu_len,
        length_ratio,
        overlap
    ])


train_text_features = text_features(
    ref_train,
    stu_train
)

test_text_features = text_features(
    ref_test,
    stu_test
)


X1_train = np.hstack([
    train_embedding_features,
    train_tfidf_features,
    train_text_features
])

X1_test = np.hstack([
    test_embedding_features,
    test_tfidf_features,
    test_text_features
])


print("\nTraining Stage 1...")

stage1_model = ExtraTreesClassifier(
    n_estimators=600,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

stage1_model.fit(
    X1_train,
    y1_train
)

stage1_predictions = stage1_model.predict(X1_test)

print("\nStage 1 Accuracy:")
print(
    round(
        accuracy_score(
            y1_test,
            stage1_predictions
        ),
        4
    )
)

print("\nStage 1 Classification Report:")

print(
    classification_report(
        y1_test,
        stage1_predictions,
        target_names=[
            "Correct",
            "Not Correct"
        ]
    )
)


stage2_mask_train = original_label[train_idx] != 0
stage2_mask_test = original_label[test_idx] != 0

X2_train = X1_train[stage2_mask_train]
X2_test = X1_test[stage2_mask_test]

stage2_original_train = original_label[
    train_idx
][stage2_mask_train]

stage2_original_test = original_label[
    test_idx
][stage2_mask_test]

y2_train = np.where(
    stage2_original_train == 2,
    0,
    1
)

y2_test = np.where(
    stage2_original_test == 2,
    0,
    1
)


print("\nTraining Stage 2...")

stage2_model = ExtraTreesClassifier(
    n_estimators=600,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

stage2_model.fit(
    X2_train,
    y2_train
)

stage2_predictions = stage2_model.predict(
    X2_test
)

print("\nStage 2 Accuracy:")

print(
    round(
        accuracy_score(
            y2_test,
            stage2_predictions
        ),
        4
    )
)

print("\nStage 2 Classification Report:")

print(
    classification_report(
        y2_test,
        stage2_predictions,
        target_names=[
            "Needs Review",
            "Likely Incorrect"
        ]
    )
)


final_predictions = []

for i in range(len(X1_test)):

    if stage1_predictions[i] == 0:
        final_predictions.append(0)

    else:
        original_index = test_idx[i]

        if original_label[original_index] == 0:
            final_predictions.append(0)
        else:
            stage2_position = np.where(
                test_idx[stage2_mask_test] == original_index
            )[0][0]

            if stage2_predictions[stage2_position] == 0:
                final_predictions.append(1)
            else:
                final_predictions.append(2)


final_predictions = np.array(
    final_predictions
)

final_actual = np.where(
    original_label[test_idx] == 0,
    0,
    np.where(
        original_label[test_idx] == 2,
        1,
        2
    )
)


print("\nFinal EZEvaluator Accuracy:")

print(
    round(
        accuracy_score(
            final_actual,
            final_predictions
        ),
        4
    )
)

print("\nFinal EZEvaluator Classification Report:")

print(
    classification_report(
        final_actual,
        final_predictions,
        labels=[0, 1, 2],
        target_names=[
            "Likely Correct",
            "Needs Review",
            "Likely Incorrect"
        ]
    )
)


joblib.dump(
    stage1_model,
    "../model/stage1_model.pkl"
)

joblib.dump(
    stage2_model,
    "../model/stage2_model.pkl"
)

joblib.dump(
    embedding_model,
    "../model/embedding_model.pkl"
)

joblib.dump(
    tfidf,
    "../model/tfidf_vectorizer.pkl"
)

print("\nModels saved successfully.")