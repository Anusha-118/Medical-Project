# Install dependencies (run once in Colab)
# !pip install -U gradio google-genai

import json
import re
import gradio as gr
from google import genai

# ---------------------------
# Put your API key here
# ---------------------------
API_KEY = "AIzaSyCgXDq5QPNAM1ku5suL-1Vg4QtXMHPnniM"

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# Fallback YouTube links (used if Gemini doesn't include links)
FALLBACK_VIDEOS = {
    "summary": "https://www.youtube.com/watch?v=JfS4y3WmT8E",  # general symptom overview
    "diet": "https://www.youtube.com/watch?v=2-5bZk1qG1k",     # diet suggestions
    "care": "https://www.youtube.com/watch?v=Vb3m6uH7r5M"      # general health care tips
}

def extract_json_from_text(text):
    """
    Try to extract the first JSON object from the model text.
    Returns dict or None.
    """
    if not text:
        return None
    # Find first { ... } block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    # Clean common ```` or ```json fences
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)
    # Attempt to load JSON
    try:
        return json.loads(candidate)
    except Exception:
        # Try to fix some common issues: replace single quotes with double quotes, trailing commas
        try:
            fixed = candidate.replace("'", '"')
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*]", "]", fixed)
            return json.loads(fixed)
        except Exception:
            return None

def safe_text_from_response(response):
    """
    Robust extraction of text from the genai response object.
    """
    if response is None:
        return ""
    # Common direct .text
    if hasattr(response, "text") and response.text:
        return response.text
    # Try candidates path (older/newer SDK shapes)
    try:
        if hasattr(response, "candidates") and response.candidates:
            # candidate may have .content.parts or .content or .text
            cand = response.candidates[0]
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                parts = cand.content.parts
                if parts and len(parts) > 0:
                    # parts may have .text or be simple strings
                    part = parts[0]
                    return getattr(part, "text", str(part))
            if hasattr(cand, "content") and isinstance(cand.content, str):
                return cand.content
            if hasattr(cand, "text"):
                return cand.text
    except Exception:
        pass
    # Last resort: string representation
    try:
        return str(response)
    except Exception:
        return ""

def medical_analyzer(symptoms, image):
    """
    Main function called by Gradio.
    symptoms: str
    image: filepath (string) or None
    """
    # Validate input
    if not symptoms and not image:
        return ("Please enter symptoms or upload an image.", "", "")

    # Build inputs for Gemini
    contents = []
    if symptoms:
        contents.append(f"Patient Symptoms: {symptoms}")
    if image:
        # image is a filepath (gr.Image with type="filepath")
        try:
            uploaded_file = client.files.upload(file=image)
            contents.append(uploaded_file)
        except Exception as e:
            # Upload failed (likely API key or network); return error message
            return (f"Image upload failed: {e}", "", "")

    # Ask Gemini to return strict JSON for reliable parsing
    prompt = """
You are a helpful, careful AI medical doctor. Analyze the patient's symptoms and optional image.
RESPOND ONLY IN JSON (no extra commentary). Produce the following JSON structure exactly:

{
  "summary_en": "<short symptom summary in English>",
  "summary_te": "<short symptom summary in Telugu>",
  "diet_en": "<short diet recommendations in English>",
  "diet_te": "<short diet recommendations in Telugu>",
  "care_en": "<short health care instructions in English>",
  "care_te": "<short health care instructions in Telugu>",
  "videos": {
    "summary": "<one YouTube URL that explains the symptoms or overview>",
    "diet": "<one YouTube URL with diet recommendations>",
    "care": "<one YouTube URL with health care instructions>"
  }
}

Keep each text concise (2-5 sentences). Use clear, non-diagnostic language and add general advice (seek in-person medical care when needed).
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents + [prompt],
        )
    except Exception as e:
        return (f"Gemini API call failed: {e}", "", "")

    raw_text = safe_text_from_response(response)

    # Try to parse JSON out of the response
    parsed = extract_json_from_text(raw_text)

    if parsed:
        # Fill fallbacks if missing
        videos = parsed.get("videos", {})
        summary_en = parsed.get("summary_en", "").strip()
        summary_te = parsed.get("summary_te", "").strip()
        diet_en = parsed.get("diet_en", "").strip()
        diet_te = parsed.get("diet_te", "").strip()
        care_en = parsed.get("care_en", "").strip()
        care_te = parsed.get("care_te", "").strip()

        vid_summary = videos.get("summary") if isinstance(videos, dict) else None
        vid_diet = videos.get("diet") if isinstance(videos, dict) else None
        vid_care = videos.get("care") if isinstance(videos, dict) else None

        if not vid_summary:
            vid_summary = FALLBACK_VIDEOS["summary"]
        if not vid_diet:
            vid_diet = FALLBACK_VIDEOS["diet"]
        if not vid_care:
            vid_care = FALLBACK_VIDEOS["care"]

        # Combine English + Telugu and append clickable link text
        summary_box = f"Summary (EN):\n{summary_en}\n\nSummary (TE):\n{summary_te}\n\nVideo: {vid_summary}"
        diet_box = f"Diet (EN):\n{diet_en}\n\nDiet (TE):\n{diet_te}\n\nVideo: {vid_diet}"
        care_box = f"Health Care (EN):\n{care_en}\n\nHealth Care (TE):\n{care_te}\n\nVideo: {vid_care}"

        return summary_box, diet_box, care_box

    # If JSON parsing failed, fallback to marker-based splitting
    text = raw_text or "No response text from Gemini."
    try:
        # Try splitting by our emoji markers
        if "🥗" in text and "💊" in text:
            s = text.split("🥗")[0].replace("🩺 Symptom Summary", "").strip()
            d = text.split("🥗")[1].split("💊")[0].strip()
            c = text.split("💊")[1].strip()
            # Append fallback videos (since we cannot reliably extract)
            s += f"\n\nVideo: {FALLBACK_VIDEOS['summary']}"
            d += f"\n\nVideo: {FALLBACK_VIDEOS['diet']}"
            c += f"\n\nVideo: {FALLBACK_VIDEOS['care']}"
            return s, d, c
        else:
            # Put full text into summary box and leave others empty
            return text, "", ""
    except Exception as e:
        return (f"Error processing response: {e}", "", "")


# ---------------------------
# Gradio UI
# ---------------------------
css = """
body {
  background: linear-gradient(135deg,#ffecd2,#fcb69f);
  font-family: 'Segoe UI', sans-serif;
  color: #111;
}
h1 {
  color:#1565c0;
  text-align:center;
  background:#bbdefb;
  padding:12px;
  border-radius:10px;
  box-shadow:0px 4px 8px rgba(0,0,0,0.12);
}
.gr-box {
  background: rgba(255,255,255,0.02);
  border-radius:10px;
  padding:10px;
  margin-bottom:12px;
  border: 1px solid rgba(255,255,255,0.04);
}
"""

with gr.Blocks(css=css, title="AI Medical Symptom Analyzer") as demo:
    gr.HTML("<h1>🧑‍⚕️ AI Medical Symptom Analyzer Chatbot</h1>")

    with gr.Row():
        symptoms = gr.Textbox(label="Enter Symptoms", placeholder="e.g. headache, fever, sore throat...")
        image = gr.Image(type="filepath", label="Upload Image (optional)")

    analyze_btn = gr.Button("🔍 Analyze")

    with gr.Row():
        summary = gr.Textbox(label="🩺 Symptom Summary (English + Telugu + Video)", lines=8)
    with gr.Row():
        diet = gr.Textbox(label="🥗 Diet Recommendations (English + Telugu + Video)", lines=6)
    with gr.Row():
        care = gr.Textbox(label="💊 Health Care Instructions (English + Telugu + Video)", lines=6)

    analyze_btn.click(fn=medical_analyzer, inputs=[symptoms, image], outputs=[summary, diet, care])

# Launch app (in Colab this will produce a public .gradio.live link)
demo.launch()
#run the code in google colab
