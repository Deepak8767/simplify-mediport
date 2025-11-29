![WhatsApp Image 2025-11-29 at 12 20 40_2f4ae1eb](https://github.com/user-attachments/assets/80989cb7-b333-4837-9346-8e49893491d9)# 🩺 MediMind – AI-Powered Medical Report Analyzer
✨ Agentic AI | OCR | Medical Understanding | Multilingual | Medication | Billing | Visualization

MediMind is an Agentic AI healthcare assistant that analyzes medical reports, prescriptions, scanned documents, and hospital bills — and explains them in simple, patient-friendly language.
It supports multilingual translation (Hindi, Marathi, Tamil, Telugu, Kannada, Bengali, Gujarati, Punjabi, Urdu) and doctor-style explanations.

# 🚀 Features
# 🧠 Agentic AI Medical Pipeline

The app uses multiple AI agents to analyze medical documents:

Summary Agent – Doctor-style patient summary

Findings Agent – Abnormal & normal parameters

Risk Agent – Low/Moderate/High risk evaluation

Diet Agent – Food recommendations + 1-day plan

Precautions Agent – Do’s, Don’ts & emergency conditions

Doctor Questions Agent – Questions patients should ask

Specialist Agent – Which doctor to consult

Next-Steps Agent – Follow-up tests & lifestyle actions

🏥 Hospital Bill Analyzer

For documents detected as Hospital Bills, the app provides:

Easy explanation of billing terms

Charge category breakdown

Percentage analysis

Insurance usage explanation

Recommended next steps & dispute guide

# 📄 OCR + Scanned Document Support

# Supports:

PDF (digital or scanned)

JPG / PNG images

Multi-page PDFs

PDF fallback → OCR using Tesseract

Accurate medical text extraction

🌐 Multilingual Support (India-Focused)

Translate output into:

Hindi (hi)

Marathi (mr)

Gujarati (gu)

Tamil (ta)

Telugu (te)

Kannada (kn)

Bengali (bn)

Punjabi (pa)

Urdu (ur)

English (en)

➡ Uses strict forced translation to prevent Gemini from returning English.

# 📊 Medical Data Visualization

# Automatically extracts:

Hemoglobin

WBC

Platelets

Any numeric lab parameter

Then generates:

Bar chart + normal range visualization

Graph is returned as base64 and shown in the frontend.

# 🔍 Auto Document Type Classification

AI classifier detects:

Medical Lab Report

Radiology Report

Prescription

Doctor Notes

Hospital Bill

Insurance Form

Unknown document

# 🛠 Tech Stack
Backend

Python

Flask

Gemini 2.0 models

PyPDF2

Tesseract OCR

pdf2image

ReportLab (PDF export)

Frontend

HTML / JS

Markdown → HTML conversion

# 📦 Installation
1. Clone the Repository
git clone https://github.com/yourusername/MediMind.git
cd MediMind

2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

3. Install Requirements
pip install -r requirements.txt

4. Create .env
GENAI_API_KEY=YOUR_API_KEY
FLASK_SECRET=YOUR_SECRET_KEY

▶ Run the Application
python app.py


# Visit:
👉 http://127.0.0.1:5000

# 📤 API Routes Used
/upload

Upload file → start background analysis task.

/status/<task_id>

Check extraction & analysis progress.

/result/<task_id>

Receive HTML summary + visualization.

/download/<task_id>

Download PDF report.

# 📁 Project Structure
MediMind/
│── static/

│── templates/

│   └── index.html

│── fonts/

│   └── (Noto Sans fonts for Indic languages)

│── app.py

│── requirements.txt

│── README.md

│── .env

# 🧪 Future Enhancements

Medication interaction checker

Symptom-based disease prediction

Voice-based doctor explanation

Chat with your report

Mobile app version

# 🤝 Contributing

Pull requests are welcome!
If you find bugs, create an issue.
