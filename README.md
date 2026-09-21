# EZEvaluator — AI-Assisted Answer Paper Evaluation System

EZEvaluator is an AI-assisted answer paper evaluation system designed to help teachers analyze handwritten student answer papers more efficiently.

The system accepts a question paper, teacher answer key, and handwritten student answer paper as PDF files. Handwritten content is extracted using AI-based document understanding, after which a machine learning model compares student answers with the corresponding reference answers and classifies them into three categories:

- Likely Correct
- Needs Teacher Review
- Likely Incorrect

EZEvaluator is designed as an assistive system. The final evaluation decision remains with the teacher.


## Key Features

- Upload question paper, answer key, and handwritten answer paper as PDFs
- Extract text from document pages using AI-based document understanding
- Match questions, reference answers, and student answers
- Compare answers using semantic and textual features
- Machine learning based answer classification
- Three-level evaluation output:
  - Likely Correct
  - Needs Teacher Review
  - Likely Incorrect
- Supports multiple questions in a single evaluation
- Simple browser-based interface
- Teacher-in-the-loop evaluation workflow


## System Architecture

Question Paper PDF
        |
        v
Document Understanding / OCR
        |
        +------------------+
        |                  |
        v                  v
Questions          Reference Answers
        |                  |
        +---------+--------+
                  |
                  v
       Student Answer Paper PDF
                  |
                  v
       Student Answer Extraction
                  |
                  v
          Feature Extraction
                  |
                  +-- Semantic Similarity
                  +-- TF-IDF Similarity
                  +-- Keyword Overlap
                  +-- Answer Length Features
                  |
                  v
         Machine Learning Models
                  |
                  v
          Answer Classification
                  |
             +----+----+
             |    |    |
             v    v    v
          Correct Review Incorrect


## Machine Learning Approach

The answer classification component uses a two-stage machine learning pipeline.

### Stage 1 — Correct vs Not Correct

The first model identifies whether an answer is likely correct or requires further analysis.

### Stage 2 — Review vs Incorrect

Answers identified as not clearly correct are passed to a second model that distinguishes between:

- Needs Teacher Review
- Likely Incorrect

This two-stage approach was used to better separate clearly correct answers from answers that require further evaluation.


## Feature Engineering

The model combines multiple types of features:

- Sentence embeddings using all-MiniLM-L6-v2
- Semantic similarity between question, reference answer, and student answer
- TF-IDF cosine similarity
- Reference/student answer length
- Answer length ratio
- Word overlap between reference and student answers
- Pairwise relationships between question, reference answer, and student answer embeddings

The final classifiers use ExtraTreesClassifier.


## Model Performance

The answer classification model was evaluated on a held-out test split of the SciEntsBank dataset.

Accuracy: 82.49%

Macro F1 Score: 0.80

Per-class F1 scores:

- Likely Correct: 0.92
- Needs Teacher Review: 0.72
- Likely Incorrect: 0.77

These results represent the performance of the answer-classification component on the evaluation dataset. They do not represent the accuracy of the complete OCR/document-processing pipeline.


## Technology Stack

### Backend

- Python
- Flask
- Google Gemini API
- PyMuPDF
- python-dotenv

### Machine Learning

- scikit-learn
- Sentence Transformers
- Extra Trees Classifier
- TF-IDF
- NumPy
- Joblib

### Frontend

- HTML
- CSS
- JavaScript
- Fetch API

### Development and Deployment

- Git
- GitHub
- Python Virtual Environment
- Render


## Project Structure

EZEvaluator/
|
+-- model/
|   +-- embedding_model.pkl
|   +-- stage1_model.pkl
|   +-- stage2_model.pkl
|   +-- tfidf_vectorizer.pkl
|
+-- static/
|   +-- script.js
|   +-- style.css
|
+-- templates/
|   +-- index.html
|
+-- training/
|   +-- inspect_data.py
|   +-- test_prediction.py
|   +-- train_model.py
|   +-- train-00001.parquet
|
+-- app.py
+-- requirements.txt
+-- .gitignore
+-- README.md


## Dataset

The answer-classification model was trained using the SciEntsBank dataset available through Hugging Face.

Dataset:

nkazi/SciEntsBank

The dataset contains question, reference answer, student answer, and evaluation label information.

The original labels were transformed into the three application-level categories used by EZEvaluator:

Original Dataset:

- Correct
- Partially Correct
- Incorrect
- Irrelevant
- No Meaningful Answer

EZEvaluator Classes:

- Likely Correct
- Needs Teacher Review
- Likely Incorrect


## How It Works

1. The user uploads the question paper.
2. The user uploads the teacher's answer key.
3. The user uploads the student's handwritten answer paper.
4. The documents are processed using AI-based document understanding.
5. Questions and corresponding answers are extracted.
6. Student answers are paired with their reference answers.
7. Semantic and textual features are generated.
8. The trained machine learning pipeline evaluates each answer.
9. Results are displayed as:
   - Likely Correct
   - Needs Teacher Review
   - Likely Incorrect


## Local Setup

### 1. Clone the repository

git clone https://github.com/Sourav99477/EZEvaluator.git

cd EZEvaluator


### 2. Create a virtual environment

Windows:

python -m venv .venv

.venv\Scripts\activate

Linux/macOS:

python3 -m venv .venv

source .venv/bin/activate


### 3. Install dependencies

pip install -r requirements.txt


### 4. Configure the Gemini API key

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Do not commit the .env file to GitHub.


### 5. Run the application

python app.py

Open the application in your browser:

http://127.0.0.1:5000


## Example Evaluation

For each student answer, EZEvaluator produces a classification such as:

Question 1 -> Likely Correct
Question 2 -> Needs Teacher Review
Question 3 -> Likely Incorrect
Question 4 -> Likely Correct


## Design Considerations

### Teacher-in-the-Loop

EZEvaluator does not replace the teacher's final judgment.

Answers classified as uncertain are explicitly placed into the Needs Teacher Review category so that the teacher can make the final decision.

### Separation of Responsibilities

The system separates document processing from answer classification:

AI Document Understanding
        |
        v
Text Extraction
        |
        v
Feature Engineering
        |
        v
Machine Learning Classification
        |
        v
Teacher Review


## Limitations

- Handwritten text quality can affect extraction accuracy.
- Complex mathematical notation and diagrams may require additional processing.
- The classification model is trained on SciEntsBank and may not generalize equally well to every academic subject.
- The reported 82.49% accuracy applies to the answer-classification component on the held-out dataset, not the complete end-to-end system.
- AI document processing depends on API availability and applicable usage limits.
- Final academic evaluation should remain under teacher supervision.


## Future Improvements

- Subject-specific evaluation models
- Better handwriting and mathematical expression recognition
- Question-wise marks prediction
- Detailed feedback generation for students
- Confidence scores for each prediction
- Support for diagrams and equations
- Teacher correction and feedback interface
- Batch processing of multiple answer papers
- Model fine-tuning on institution-specific evaluation data


## Project Objective

The goal of EZEvaluator is to demonstrate how document AI, natural language processing, feature engineering, and machine learning can be combined to build a practical educational automation system.

Rather than fully automating academic evaluation, the system focuses on assisting teachers by identifying answers that are likely correct, likely incorrect, or require human review.


## Author

Sourav R

GitHub:
https://github.com/Sourav99477