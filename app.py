import os
import re
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from sklearn.preprocessing import normalize

load_dotenv()

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

embedding_model = joblib.load("model/embedding_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
stage1_model = joblib.load("model/stage1_model.pkl")
stage2_model = joblib.load("model/stage2_model.pkl")


def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_answer(text):
    text = clean_text(text)

    text = re.sub(
        r"^\s*\d+\s*[\.\):\-]\s*",
        "",
        text
    )

    text = re.sub(
        r"^\s*[-•]\s*",
        "",
        text
    )

    return text.strip()


def extract_questions(text):
    text = clean_text(text)

    matches = re.split(
        r"(?=QUESTION\s+\d+\s*:)",
        text,
        flags=re.IGNORECASE
    )

    result = {}

    for block in matches:

        match = re.match(
            r"QUESTION\s+(\d+)\s*:\s*(.*)",
            block,
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            continue

        number = int(match.group(1))
        content = match.group(2).strip()

        if content:
            result[number] = content

    return result


def extract_reference_answers(text):
    text = clean_text(text)

    matches = re.split(
        r"(?=REFERENCE\s+ANSWER\s+\d+\s*:)",
        text,
        flags=re.IGNORECASE
    )

    result = {}

    for block in matches:

        match = re.match(
            r"REFERENCE\s+ANSWER\s+(\d+)\s*:\s*(.*)",
            block,
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            continue

        number = int(match.group(1))
        content = match.group(2).strip()

        if content:
            result[number] = content

    return result


def extract_student_answers(text):
    text = clean_text(text)

    matches = re.split(
        r"(?=QUESTION\s+\d+\s+ANSWER\s*:)",
        text,
        flags=re.IGNORECASE
    )

    result = {}

    for block in matches:

        match = re.match(
            r"QUESTION\s+(\d+)\s+ANSWER\s*:\s*(.*)",
            block,
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            continue

        number = int(match.group(1))
        content = match.group(2).strip()

        content = clean_answer(content)

        if content:
            result[number] = content

    return result


def build_embedding_features(q, ref, stu):

    q_ref = np.abs(q - ref)
    q_stu = np.abs(q - stu)
    ref_stu = np.abs(ref - stu)

    q_ref_product = q * ref
    q_stu_product = q * stu
    ref_stu_product = ref * stu

    q_ref_cosine = np.sum(
        q * ref,
        axis=1,
        keepdims=True
    )

    q_stu_cosine = np.sum(
        q * stu,
        axis=1,
        keepdims=True
    )

    ref_stu_cosine = np.sum(
        ref * stu,
        axis=1,
        keepdims=True
    )

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

        r_words = set(
            r.lower().split()
        )

        s_words = set(
            s.lower().split()
        )

        if not r_words:
            overlap.append(0)
        else:
            overlap.append(
                len(r_words & s_words)
                / len(r_words)
            )

    return np.column_stack([
        ref_len,
        stu_len,
        length_ratio,
        overlap
    ])


def predict_answers(
    questions,
    references,
    students
):

    questions = np.array(questions)
    references = np.array(references)
    students = np.array(students)

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

    q_tfidf = normalize(
        tfidf.transform(questions)
    )

    ref_tfidf = normalize(
        tfidf.transform(references)
    )

    stu_tfidf = normalize(
        tfidf.transform(students)
    )

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

    results = []

    for i in range(len(students)):

        results.append({
            "question_number": i + 1,
            "question": questions[i],
            "answer": students[i],
            "label": labels[
                final_predictions[i]
            ]
        })

    return results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():

    required_files = [
        "question_pdf",
        "key_pdf",
        "answer_pdf"
    ]

    for field in required_files:

        if field not in request.files:

            return jsonify({
                "success": False,
                "error": f"{field} is missing."
            }), 400

        file = request.files[field]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": f"No file selected for {field}."
            }), 400

        if not file.filename.lower().endswith(".pdf"):

            return jsonify({
                "success": False,
                "error": f"{field} must be a PDF."
            }), 400

    os.makedirs(
        "temp",
        exist_ok=True
    )

    question_path = "temp/question_paper.pdf"
    key_path = "temp/answer_key.pdf"
    answer_path = "temp/answer_paper.pdf"

    request.files[
        "question_pdf"
    ].save(question_path)

    request.files[
        "key_pdf"
    ].save(key_path)

    request.files[
        "answer_pdf"
    ].save(answer_path)

    try:

        print("\nUploading question paper...")

        question_pdf = client.files.upload(
            file=question_path
        )

        print("Uploading answer key...")

        key_pdf = client.files.upload(
            file=key_path
        )

        print("Uploading handwritten answer paper...")

        answer_pdf = client.files.upload(
            file=answer_path
        )

        print("\nRunning ONE Gemini request...")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                question_pdf,
                key_pdf,
                answer_pdf,
                """
You are processing three documents for an answer evaluation system.

DOCUMENT 1:
Question paper.

DOCUMENT 2:
Teacher answer key.

DOCUMENT 3:
Student handwritten answer paper.

Match everything using the original question numbers.

Extract the questions, reference answers, and student answers.

Return ONLY this format:

QUESTION 1:
<question text>

REFERENCE ANSWER 1:
<teacher reference answer>

QUESTION 1 ANSWER:
<student answer>

QUESTION 2:
<question text>

REFERENCE ANSWER 2:
<teacher reference answer>

QUESTION 2 ANSWER:
<student answer>

QUESTION 3:
<question text>

REFERENCE ANSWER 3:
<teacher reference answer>

QUESTION 3 ANSWER:
<student answer>

Continue for every question.

Important rules:

Preserve the original numbering.

Do not move an answer to another question.

Do not include question numbers such as "1)", "2)", or "3.)" inside the student answer.

Do not summarize.

Do not evaluate the student's answers.

Do not invent missing text.

If handwriting contains minor spelling errors, preserve what the student actually wrote.
"""
            ]
        )

        extracted_text = response.text

        print(
            "\nGEMINI RESULT\n"
        )

        print(extracted_text)

        question_map = extract_questions(
            extracted_text
        )

        reference_map = extract_reference_answers(
            extracted_text
        )

        answer_map = extract_student_answers(
            extracted_text
        )

        print(
            "\nQUESTIONS\n"
        )

        print(question_map)

        print(
            "\nREFERENCES\n"
        )

        print(reference_map)

        print(
            "\nSTUDENT ANSWERS\n"
        )

        print(answer_map)

        common_numbers = sorted(
            set(question_map)
            & set(reference_map)
            & set(answer_map)
        )

        print(
            "\nMATCHED QUESTION NUMBERS"
        )

        print(common_numbers)

        if not common_numbers:

            return jsonify({
                "success": False,
                "error": (
                    "No questions could be "
                    "matched across the three documents."
                )
            }), 500

        questions = [
            question_map[number]
            for number in common_numbers
        ]

        references = [
            reference_map[number]
            for number in common_numbers
        ]

        students = [
            answer_map[number]
            for number in common_numbers
        ]

        print(
            "\nMATCHED DATA\n"
        )

        for i, number in enumerate(
            common_numbers
        ):

            print(
                f"QUESTION NUMBER: {number}"
            )

            print(
                f"QUESTION: {questions[i]}"
            )

            print(
                f"REFERENCE: {references[i]}"
            )

            print(
                f"STUDENT: {students[i]}"
            )

            print(
                "-" * 70
            )

        results = predict_answers(
            questions,
            references,
            students
        )

        for i, result in enumerate(results):

            result[
                "question_number"
            ] = common_numbers[i]

        print(
            "\nFINAL RESULTS\n"
        )

        for result in results:

            print(
                f"Question "
                f"{result['question_number']}: "
                f"{result['label']}"
            )

        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as error:

        print(
            "\nERROR"
        )

        print(error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    finally:

        for path in [
            question_path,
            key_path,
            answer_path
        ]:

            if os.path.exists(path):
                os.remove(path)


@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": (
            "Total uploaded files are too large. "
            "Maximum is 60 MB."
        )
    }), 413


if __name__ == "__main__":
    app.run(
        debug=True
    )