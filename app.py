import os
import io
import uuid
import json
import base64

import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes

from flask import Flask, request, jsonify, send_file, render_template
from concurrent.futures import ThreadPoolExecutor

import markdown as md

# Matplotlib for visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ReportLab for PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Gemini
import google.generativeai as genai
from dotenv import load_dotenv

# ------------------ INIT ------------------

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret")

# Models
REASONING_MODEL = "gemini-2.0-flash"
TRANSLATION_MODEL = "gemini-1.5-flash"

app = Flask(__name__)
app.secret_key = FLASK_SECRET

# -------- Unicode font setup for PDF --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

# Map language code -> Font Name (registered in ReportLab)
LANG_FONT_MAP = {
    "en": "NotoSans-Regular",
    "hi": "NotoSansDevanagari-Regular",
    "mr": "NotoSansDevanagari-Regular",
    "gu": "NotoSansGujarati-Regular",
    "ta": "NotoSansTamil-Regular",
    "te": "NotoSansTelugu-Regular",
    "kn": "NotoSansKannada-Regular",
    "bn": "NotoSansBengali-Regular",
    "pa": "NotoSansGurmukhi-Regular",
    "ur": "NotoNaskhArabic-Regular",
}

# Register all fonts found in the map
REGISTERED_FONTS = set()

try:
    for lang_code, font_name in LANG_FONT_MAP.items():
        if font_name in REGISTERED_FONTS:
            continue
            
        font_path = os.path.join(FONTS_DIR, font_name + ".ttf")
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            REGISTERED_FONTS.add(font_name)
            print(f"✅ Registered font: {font_name}")
        else:
            print(f"⚠️ Font file missing: {font_path}")

except Exception as e:
    print("⚠️ Error registering fonts:", e)

# -------- Gemini config --------
GENAI_ENABLED = True
if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
        print("✅ Gemini configured.")
    except Exception as e:
        print("❌ Gemini config failed:", e)
        GENAI_ENABLED = False
else:
    print("❌ GENAI_API_KEY not set in .env")
    GENAI_ENABLED = False

EXECUTOR = ThreadPoolExecutor(max_workers=3)

# task_id -> state dict
TASKS = {}


@app.route("/")
def home():
    return render_template("index.html")


# ------------------ GEMINI HELPERS ------------------

def call_gemini(prompt: str) -> str:
    """Basic Gemini call for reasoning (doctor-style answers)."""
    if not GENAI_ENABLED:
        return "Gemini API is not configured. Please set GENAI_API_KEY."

    try:
        model = genai.GenerativeModel(REASONING_MODEL)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or "No response from Gemini."
    except Exception as e:
        return f"Gemini Error: {e}"


def translate_with_gemini(text: str, target_lang: str) -> str:
    """
    Fully forced translation using JSON schema + system instructions.
    Ensures output is ONLY in target language.
    """

    if target_lang == "en":
        return text

    if not GENAI_ENABLED:
        return text

    CHUNK = 4000
    segments = [text[i:i+CHUNK] for i in range(0, len(text), CHUNK)]
    final_output = []

    LANG_MAP = {
        "hi": "Hindi",
        "mr": "Marathi",
        "gu": "Gujarati",
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "bn": "Bengali",
        "pa": "Punjabi",
        "ur": "Urdu",
        "en": "English"
    }

    target_name = LANG_MAP.get(target_lang, target_lang)

    model = genai.GenerativeModel(
        TRANSLATION_MODEL,
        system_instruction=f"""
You are a translation engine. 
You MUST output text ONLY in {target_name}. 
Do NOT output English words except medical units like mg/dL.
Do NOT explain anything. ONLY output translation.
"""
    )

    for part in segments:
        try:
            response = model.generate_content(
                [
                    {
                        "role": "user",
                        "parts": [{
                            "text": f"""
Translate the following into **{target_name}**.

RULES:
- No English sentences allowed.
- No English headings.
- Translate EVERYTHING.
- Keep bullet points intact.

TEXT:
\"\"\"
{part}
\"\"\"
"""
                        }]
                    }
                ]
            )

            translated = getattr(response, "text", "").strip()

            # Additional safety: remove English if present
            if any(ch.isalpha() for ch in translated) and target_lang != "en":
                # Check if English dominates
                import re
                english_words = re.findall(r"[A-Za-z]+", translated)
                if len(english_words) > 5:
                    translated = f"(Translation failed, Gemini returned English.)\n{translated}"

            final_output.append(translated)

        except Exception as e:
            final_output.append(f"(Error occurred: {e})\n{part}")

    return "\n\n".join(final_output)


def markdown_to_html(md_text: str) -> str:
    try:
        return md.markdown(md_text, extensions=["extra"])
    except Exception:
        return "<pre>" + md_text + "</pre>"

# ------------------ DOCUMENT TYPE CLASSIFIER ------------------

def agent_document_classifier(extracted_text: str) -> str:
    """
    Classify document type:
    - Medical Lab Report
    - Radiology Report
    - Prescription
    - Doctor Notes
    - Hospital Bill
    - Insurance Form
    - Other/Unknown
    """
    prompt = f"""
You are a document classifier for healthcare.

Identify the document type from this text. Choose exactly one:

- Medical Lab Report
- Radiology Report
- Prescription
- Doctor Notes
- Hospital Bill
- Insurance Form
- Other/Unknown

Return ONLY the label, nothing else.

Text:
{extracted_text[:4000]}
"""
    resp = call_gemini(prompt).strip()
    for choice in [
        "Medical Lab Report",
        "Radiology Report",
        "Prescription",
        "Doctor Notes",
        "Hospital Bill",
        "Insurance Form",
        "Other/Unknown",
    ]:
        if choice.lower() in resp.lower():
            return choice
    return "Other/Unknown"


# ------------------ MULTI-AGENT MEDICAL PIPELINE ------------------

def explanation_style_text(level: str) -> str:
    if level == "simple":
        return "Explain as if to a 12-year-old child, using very simple language."
    elif level == "detailed":
        return "Explain in detailed but clear language, suitable for a medical student."
    return "Explain in normal, clear language for an educated non-medical adult."


def common_context(extracted_text: str, symptoms: str) -> str:
    extra = ""
    if symptoms:
        extra = f"\nThe patient also reports the following symptoms in their own words:\n{symptoms}\n"
    return f"Medical report text:\n{extracted_text}\n{extra}"


def agent_summary(extracted_text: str, explanation_level: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 1: Medical Summary Specialist, speaking like a caring doctor.

{explanation_style_text(explanation_level)}

Summarize the report in a way that a patient can understand.
Focus on what is happening in the patient's body.

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_findings(extracted_text: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 2: Findings Detector (doctor-level reasoning, patient-friendly explanation).

From the report, list:
- Key abnormal values
- What these might indicate
- Any important normal values

Use Markdown:

### Key Findings
- ...

### Possible Issues
- ...

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_diet_recommendations(extracted_text: str, explanation_level: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 3: Diet & Lifestyle Coach (doctor + nutritionist style).

{explanation_style_text(explanation_level)}

Based on the report and symptoms, suggest:
- Foods to include
- Foods to avoid
- A simple 1-day sample diet plan
- Mention if the patient should avoid alcohol or smoking.

Use bullet points and be realistic for Indian context if applicable.

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_precautions(extracted_text: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 4: Precaution Advisor (doctor speaking to patient).

List:
- Do's (what patient should do)
- Don'ts (what to avoid)
- When the patient should seek urgent medical help

Use clear headings and bullet points.

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_risk_score(extracted_text: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 5: Risk Evaluator.

Estimate:
- Overall risk level: Low / Moderate / High
- 2–3 lines explaining why (like a doctor explaining risk, NOT giving final diagnosis).

Format:

### Risk Level
- Overall: <Low/Moderate/High>

### Reason
- ...

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_questions_for_doctor(extracted_text: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 6: Patient Helper.

Generate 5–7 simple questions the patient can ask their doctor based on this report and symptoms.

Format:

### Questions to Ask Your Doctor
1. ...
2. ...
3. ...

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_condition_and_doctor(extracted_text: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 7: Condition & Specialist Advisor.

From this report and symptoms:
- List possible conditions or problem areas (NOT a confirmed diagnosis).
- Suggest what type of doctor/specialist they should consult (e.g., general physician, cardiologist, endocrinologist).

Format:

### Possible Conditions (Not a Final Diagnosis)
- ...

### Recommended Doctor Type
- ...

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_next_steps_checklist(extracted_text: str, symptoms: str) -> str:
    prompt = f"""
You are Agent 8: Next Steps Planner (doctor explaining next actions).

Create a checklist:
- Tests to repeat or follow up
- Lifestyle changes (sleep, stress, exercise)
- Monitoring points at home
- When to book follow-up

Format:

### Next Steps for You
- ...

{common_context(extracted_text, symptoms)}
"""
    return call_gemini(prompt)


def agent_extract_values(extracted_text: str) -> dict:
    """
    Agent 9: extract numeric lab values + normal ranges.
    """
    prompt = f"""
You are Agent 9: Lab Value Extractor.

From the medical report text, extract numeric lab values.

Return ONLY valid JSON (no extra text) in this format:

{{
  "Hemoglobin": {{"value": 11.2, "normal_low": 13, "normal_high": 17}},
  "WBC": {{"value": 7600, "normal_low": 4000, "normal_high": 11000}},
  "Platelets": {{"value": 150000, "normal_low": 150000, "normal_high": 450000}}
}}

Use null if normal_low or normal_high are unknown.

Report text:
{extracted_text}
"""
    resp = call_gemini(prompt)

    try:
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1:
            json_str = resp[start:end + 1]
        else:
            json_str = resp
        return json.loads(json_str)
    except Exception:
        return {}


# ------------------ BILLING PIPELINE ------------------

def agent_billing_explainer(extracted_text: str) -> str:
    prompt = f"""
You are a hospital doctor who understands billing and wants to help the patient.

Explain this hospital bill in very simple terms:
- Why the patient was charged
- What main services or procedures were done
- What "patient responsibility" means
- How insurance was applied (if present)
- What to do if the bill feels too high
- What support options exist (payment plan, financial assistance)

Use friendly, non-threatening language.
Use bullet points and short paragraphs.

Bill text:
{extracted_text}
"""
    return call_gemini(prompt)


def agent_billing_breakdown(extracted_text: str) -> str:
    prompt = f"""
You are a billing analyst.

Break down this hospital bill:

- Key charge categories (e.g., Cardiology, Lab, Pharmacy)
- Approximate proportion of total each category represents
- Any unusual or very high-cost items

Format in Markdown:

### Charge Categories
- Category – approx amount – approx % of total

### Notes
- ...

Bill text:
{extracted_text}
"""
    return call_gemini(prompt)


def agent_billing_next_steps(extracted_text: str) -> str:
    prompt = f"""
You are a financial counselor at the hospital.

Suggest next steps for the patient:

- How to confirm if the bill is correct
- How to talk to billing/insurance
- How to request an itemized bill (if not provided)
- How to ask for a discount or payment plan
- When to seek help from a social worker or financial counselor

Use bullet points and simple language.

Bill text:
{extracted_text}
"""
    return call_gemini(prompt)


# ------------------ PIPELINE ORCHESTRATORS ------------------

def run_medical_pipeline(extracted_text: str, language: str,
                         explanation_level: str, symptoms: str) -> dict:
    """Full medical agentic pipeline."""
    summary_md = agent_summary(extracted_text, explanation_level, symptoms)
    findings_md = agent_findings(extracted_text, symptoms)
    diet_md = agent_diet_recommendations(extracted_text, explanation_level, symptoms)
    precautions_md = agent_precautions(extracted_text, symptoms)
    risk_md = agent_risk_score(extracted_text, symptoms)
    questions_md = agent_questions_for_doctor(extracted_text, symptoms)
    condition_md = agent_condition_and_doctor(extracted_text, symptoms)
    next_steps_md = agent_next_steps_checklist(extracted_text, symptoms)

    english_md = f"""
# Medical Report – Agentic Doctor-Style Analysis

## 1. Summary (Doctor Explains Your Report)
{summary_md}

## 2. Key Findings & Possible Issues
{findings_md}

## 3. Risk Evaluation
{risk_md}

## 4. Possible Conditions & Doctor Type
{condition_md}

## 5. Diet & Lifestyle Recommendations
{diet_md}

## 6. Precautions (Do's & Don'ts)
{precautions_md}

## 7. Questions to Ask Your Doctor
{questions_md}

## 8. Next Steps Checklist
{next_steps_md}
""".strip()

    print(f"DEBUG [run_medical_pipeline]: Language requested = {language}")
    
    translated_md = translate_with_gemini(english_md, language)
    
    print(f"DEBUG [run_medical_pipeline]: Translation complete")

    return {
        "english_md": english_md,
        "translated_md": translated_md,
    }


def run_billing_pipeline(extracted_text: str, language: str) -> dict:
    """Pipeline for hospital bills."""
    explainer_md = agent_billing_explainer(extracted_text)
    breakdown_md = agent_billing_breakdown(extracted_text)
    next_steps_md = agent_billing_next_steps(extracted_text)

    english_md = f"""
# Hospital Bill – Doctor-style Explanation

## 1. What This Bill Means
{explainer_md}

## 2. Breakdown of Charges
{breakdown_md}

## 3. What You Can Do Next
{next_steps_md}
""".strip()

    print(f"DEBUG [run_billing_pipeline]: Language requested = {language}")
    
    translated_md = translate_with_gemini(english_md, language)
    
    print(f"DEBUG [run_billing_pipeline]: Translation complete")

    return {
        "english_md": english_md,
        "translated_md": translated_md,
    }


# ------------------ VISUALIZATION ------------------

def generate_visualization(value_data: dict) -> str:
    """Generate bar + normal range chart, return base64 PNG or None."""
    if not value_data:
        return None

    labels = []
    actual_values = []
    normal_low = []
    normal_high = []

    for key, item in value_data.items():
        if not isinstance(item, dict):
            continue
        val = item.get("value")
        if val is None:
            continue
        labels.append(key)
        actual_values.append(val)
        normal_low.append(item.get("normal_low"))
        normal_high.append(item.get("normal_high"))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(labels))

    ax.bar(x, actual_values, label="Actual Value")

    for i, (lo, hi) in enumerate(zip(normal_low, normal_high)):
        if lo is not None and hi is not None:
            ax.plot([i, i], [lo, hi], linewidth=6)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Lab Parameters vs Normal Range")
    ax.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


# ------------------ BACKGROUND TASK ------------------

def process_file_background(task_id: str, raw: bytes, filename: str, language: str,
                            explanation_level: str, symptoms: str):
    """OCR + auto-doc-type + branch to medical/billing pipeline + visualization."""
    try:
        TASKS[task_id]["status"] = "extracting"
        TASKS[task_id]["progress"] = 15

        extracted_text = ""

        # ---- PDF ----
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(raw))
                pages = [page.extract_text() or "" for page in reader.pages]
                extracted_text = "\n".join(pages).strip()
            except Exception:
                extracted_text = ""

            # OCR fallback for scanned PDF
            if not extracted_text.strip():
                TASKS[task_id]["status"] = "ocr"
                TASKS[task_id]["progress"] = 35
                images = convert_from_bytes(raw, dpi=150)
                extracted_text = ""
                for img in images[:6]:
                    extracted_text += pytesseract.image_to_string(
                        img, config="--oem 3 --psm 6"
                    ) + "\n"

        # ---- IMAGE (scanned copy, jpg/png) ----
        else:
            TASKS[task_id]["status"] = "ocr"
            TASKS[task_id]["progress"] = 35
            img = Image.open(io.BytesIO(raw))
            extracted_text = pytesseract.image_to_string(
                img, config="--oem 3 --psm 6"
            )

        TASKS[task_id]["extracted"] = extracted_text

        # ---- Document Type Classification ----
        TASKS[task_id]["status"] = "classifying"
        TASKS[task_id]["progress"] = 50

        doc_type = agent_document_classifier(extracted_text)
        TASKS[task_id]["doc_type"] = doc_type

        # ---- Branching ----
        if doc_type == "Hospital Bill":
            TASKS[task_id]["status"] = "billing_analysis"
            TASKS[task_id]["progress"] = 70

            pipe_res = run_billing_pipeline(extracted_text, language)
            TASKS[task_id]["summary_english"] = pipe_res["english_md"]
            TASKS[task_id]["summary"] = pipe_res["translated_md"]

            TASKS[task_id]["values"] = None
            TASKS[task_id]["visualization"] = None

        else:
            TASKS[task_id]["status"] = "medical_analysis"
            TASKS[task_id]["progress"] = 65

            pipe_res = run_medical_pipeline(extracted_text, language, explanation_level, symptoms)
            TASKS[task_id]["summary_english"] = pipe_res["english_md"]
            TASKS[task_id]["summary"] = pipe_res["translated_md"]

            TASKS[task_id]["status"] = "visualizing"
            TASKS[task_id]["progress"] = 80

            values = agent_extract_values(extracted_text)
            graph_b64 = generate_visualization(values)

            TASKS[task_id]["values"] = values
            TASKS[task_id]["visualization"] = graph_b64

        TASKS[task_id]["status"] = "done"
        TASKS[task_id]["progress"] = 100

    except Exception as e:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)


# ------------------ API ROUTES ------------------

@app.route("/upload", methods=["POST"])
def upload():
    if "report" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["report"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    raw = file.read()
    filename = file.filename.lower()

    language = request.form.get("language", "en")
    explanation_level = request.form.get("explanation_level", "normal")
    symptoms = (request.form.get("symptoms") or "").strip()

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "progress": 0,
        "summary": "",
        "summary_english": "",
        "extracted": "",
        "visualization": None,
        "values": None,
        "language": language,
        "explanation_level": explanation_level,
        "symptoms": symptoms,
        "doc_type": None,
    }

    EXECUTOR.submit(
        process_file_background,
        task_id,
        raw,
        filename,
        language,
        explanation_level,
        symptoms,
    )

    return jsonify({"task_id": task_id})


@app.route("/status/<task_id>")
def status(task_id):
    t = TASKS.get(task_id)
    if not t:
        return jsonify({"error": "Invalid task ID"}), 404
    return jsonify(t)


@app.route("/result/<task_id>")
def result(task_id):
    t = TASKS.get(task_id)
    if not t:
        return jsonify({"error": "Invalid task ID"}), 404
    if t["status"] != "done":
        return jsonify({"error": "Task not completed"}), 400

    html_summary = markdown_to_html(t["summary"])

    return jsonify({
        "summary_html": html_summary,
        "summary_md": t["summary"],
        "summary_english_md": t.get("summary_english", ""),
        "extracted_text": t["extracted"],
        "language": t["language"],
        "visualization": t.get("visualization"),
        "values": t.get("values"),
        "doc_type": t.get("doc_type", "Other/Unknown"),
    })


@app.route("/download/<task_id>")
def download_pdf(task_id):
    t = TASKS.get(task_id)
    if not t or t["status"] != "done":
        return "Task not ready", 400

    content = t["summary"]
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    
    # Select font based on language
    lang = t.get("language", "en")
    font_name = LANG_FONT_MAP.get(lang, "NotoSans-Regular")
    
    # Fallback if font not registered
    if font_name not in REGISTERED_FONTS:
        # Try NotoSans-Regular as generic fallback
        if "NotoSans-Regular" in REGISTERED_FONTS:
            font_name = "NotoSans-Regular"
        else:
            font_name = "Helvetica" # Last resort

    normal_style.fontName = font_name
    normal_style.leading = 16
    
    # Handle RTL for Urdu if needed (basic support)
    if lang == "ur":
        normal_style.alignment = 2 # TA_RIGHT

    elements = []
    for line in content.split("\n"):
        elements.append(Paragraph(line, normal_style))
        elements.append(Spacer(1, 8))

    # Only embed visualization if exists (for medical reports)
    if t.get("visualization"):
        try:
            img_data = base64.b64decode(t["visualization"])
            img_buf = io.BytesIO(img_data)
            elements.append(Spacer(1, 12))
            elements.append(RLImage(img_buf, width=400, height=250))
        except Exception:
            pass

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="ai_summary.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
