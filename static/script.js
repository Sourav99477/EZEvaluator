const questionInput = document.getElementById("questionInput");
const keyInput = document.getElementById("keyInput");
const answerInput = document.getElementById("answerInput");

const questionDropZone = document.getElementById("questionDropZone");
const keyDropZone = document.getElementById("keyDropZone");
const answerDropZone = document.getElementById("answerDropZone");

const questionPreview = document.getElementById("questionPreview");
const keyPreview = document.getElementById("keyPreview");
const answerPreview = document.getElementById("answerPreview");

const questionFileName = document.getElementById("questionFileName");
const keyFileName = document.getElementById("keyFileName");
const answerFileName = document.getElementById("answerFileName");

const questionFileSize = document.getElementById("questionFileSize");
const keyFileSize = document.getElementById("keyFileSize");
const answerFileSize = document.getElementById("answerFileSize");

const removeQuestion = document.getElementById("removeQuestion");
const removeKey = document.getElementById("removeKey");
const removeAnswer = document.getElementById("removeAnswer");

const evaluateBtn = document.getElementById("evaluateBtn");

const resultsPanel = document.getElementById("resultsPanel");
const resultsList = document.getElementById("resultsList");

let questionFile = null;
let keyFile = null;
let answerFile = null;

questionInput.addEventListener("change", () => {
    if (questionInput.files.length > 0) {
        handleFile(questionInput.files[0], "question");
    }
});

keyInput.addEventListener("change", () => {
    if (keyInput.files.length > 0) {
        handleFile(keyInput.files[0], "key");
    }
});

answerInput.addEventListener("change", () => {
    if (answerInput.files.length > 0) {
        handleFile(answerInput.files[0], "answer");
    }
});

setupDropZone(questionDropZone, "question");
setupDropZone(keyDropZone, "key");
setupDropZone(answerDropZone, "answer");

function setupDropZone(zone, type) {
    zone.addEventListener("dragover", (event) => {
        event.preventDefault();
        zone.classList.add("dragover");
    });

    zone.addEventListener("dragleave", () => {
        zone.classList.remove("dragover");
    });

    zone.addEventListener("drop", (event) => {
        event.preventDefault();
        zone.classList.remove("dragover");

        if (event.dataTransfer.files.length > 0) {
            handleFile(event.dataTransfer.files[0], type);
        }
    });
}

function handleFile(file, type) {
    if (file.type !== "application/pdf") {
        alert("Please select a PDF file.");
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        alert("File size must be less than 20 MB.");
        return;
    }

    if (type === "question") {
        questionFile = file;
        questionFileName.textContent = file.name;
        questionFileSize.textContent = formatFileSize(file.size);
        questionPreview.style.display = "flex";
    }

    if (type === "key") {
        keyFile = file;
        keyFileName.textContent = file.name;
        keyFileSize.textContent = formatFileSize(file.size);
        keyPreview.style.display = "flex";
    }

    if (type === "answer") {
        answerFile = file;
        answerFileName.textContent = file.name;
        answerFileSize.textContent = formatFileSize(file.size);
        answerPreview.style.display = "flex";
    }
}

removeQuestion.addEventListener("click", () => {
    questionFile = null;
    questionInput.value = "";
    questionPreview.style.display = "none";
});

removeKey.addEventListener("click", () => {
    keyFile = null;
    keyInput.value = "";
    keyPreview.style.display = "none";
});

removeAnswer.addEventListener("click", () => {
    answerFile = null;
    answerInput.value = "";
    answerPreview.style.display = "none";
});

evaluateBtn.addEventListener("click", async () => {

    if (!questionFile) {
        alert("Please upload the question paper.");
        return;
    }

    if (!keyFile) {
        alert("Please upload the teacher answer key.");
        return;
    }

    if (!answerFile) {
        alert("Please upload the handwritten answer paper.");
        return;
    }

    const formData = new FormData();

    formData.append("question_pdf", questionFile);
    formData.append("key_pdf", keyFile);
    formData.append("answer_pdf", answerFile);

    evaluateBtn.disabled = true;
    evaluateBtn.innerHTML = "Evaluating...";

    resultsPanel.style.display = "none";
    resultsList.innerHTML = "";

    try {

        const response = await fetch("/evaluate", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Evaluation failed.");
        }

        displayResults(result.results);

    } catch (error) {
        alert(error.message);
    }

    evaluateBtn.disabled = false;
    evaluateBtn.innerHTML = 'Evaluate Answer Paper <span>→</span>';
});

function displayResults(results) {

    resultsList.innerHTML = "";

    results.forEach((result) => {

        const card = document.createElement("div");

        card.className = "result-card";

        let icon = "✓";

        if (result.label === "Needs Teacher Review") {
            icon = "?";
        }

        if (result.label === "Likely Incorrect") {
            icon = "!";
        }

        card.innerHTML = `
            <div class="result-number">
                Q${result.question_number}
            </div>

            <div class="result-content">
                <strong>${result.label}</strong>
                <p>${result.answer}</p>
            </div>

            <div class="result-status">
                ${icon}
            </div>
        `;

        resultsList.appendChild(card);
    });

    resultsPanel.style.display = "block";

    resultsPanel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

function formatFileSize(bytes) {

    if (bytes < 1024 * 1024) {
        return Math.round(bytes / 1024) + " KB";
    }

    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}