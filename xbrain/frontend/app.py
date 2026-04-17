"""
frontend/app.py
Designed Streamlit frontend for X-Brain
"""

import io
import base64
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="X-Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0b1220; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.title-box {
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.title-main {
    font-size: 2rem;
    font-weight: 800;
    color: #e5f0ff;
    margin-bottom: 4px;
}
.title-sub {
    color: #9fb3c8;
    font-size: 0.95rem;
}

.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-left: 4px solid #3b82f6;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.metric-label {
    color: #8ea3b9;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.metric-value {
    color: #f8fbff;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 4px;
}
.metric-sub {
    color: #8ea3b9;
    font-size: 0.78rem;
    margin-top: 4px;
}

.section-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.section-title {
    color: #dcecff;
    font-size: 1.02rem;
    font-weight: 800;
    margin-bottom: 12px;
}

.soft-text {
    color: #a8bfd8;
    font-size: 0.92rem;
}

.disclaimer-box {
    background: #1a1622;
    border: 1px solid #5b2130;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 14px 16px;
    color: #f2b8c6;
    font-size: 0.85rem;
    margin-top: 14px;
}

.source-chip {
    display: inline-block;
    background: #172033;
    border: 1px solid #2b3b59;
    border-radius: 999px;
    padding: 4px 10px;
    color: #b8d1f0;
    font-size: 0.75rem;
    margin: 4px 6px 4px 0;
}

.chat-user {
    background: #18263c;
    border: 1px solid #27415f;
    border-radius: 12px;
    padding: 10px 12px;
    margin: 8px 0;
    color: #d8e8ff;
}
.chat-bot {
    background: #121826;
    border: 1px solid #253041;
    border-radius: 12px;
    padding: 10px 12px;
    margin: 8px 0;
    color: #dce8f5;
}
.small-note {
    color: #8ea3b9;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def b64_to_bytes_img(b64: str) -> bytes:
    return base64.b64decode(b64)


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_languages():
    try:
        r = requests.get(f"{API_URL}/languages", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"English": "en"}


def call_analyze(img_file, language: str):
    files = {
        "file": (img_file.name, img_file.getvalue(), img_file.type or "image/jpeg")
    }
    r = requests.post(
        f"{API_URL}/analyze",
        files=files,
        params={"language": language},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def call_qa(question: str, report_ctx: str, class_name: str, history: list, language: str):
    r = requests.post(
        f"{API_URL}/qa",
        json={
            "question": question,
            "report_context": report_ctx,
            "class_name": class_name,
            "conversation_history": history,
            "language": language,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["answer"]


def call_translate(text: str, language: str):
    r = requests.post(
        f"{API_URL}/translate",
        json={"text": text, "language": language},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["translated"]


CLASS_COLORS = {
    "glioma": "#ef4444",
    "meningioma": "#f59e0b",
    "pituitary": "#3b82f6",
    "notumor": "#22c55e",
}

URGENCY_COLORS = {
    "HIGH": "#ef4444",
    "MODERATE": "#f59e0b",
    "LOW-MODERATE": "#3b82f6",
    "NONE": "#22c55e",
}


# ── Session state ─────────────────────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_file_name" not in st.session_state:
    st.session_state.analysis_file_name = None

if "analysis_language" not in st.session_state:
    st.session_state.analysis_language = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "translated_report" not in st.session_state:
    st.session_state.translated_report = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 X-Brain")
    st.markdown("**Explainable Brain Tumor Analysis**")
    st.markdown("---")

    health = check_api_health()
    st.markdown("### System Status")
    if health:
        st.write(f"Classifier: {'✅' if health.get('classifier') else '❌'}")
        st.write(f"Segmentor: {'✅' if health.get('segmentor') else '❌'}")
        st.write(f"RAG Index: {'✅' if health.get('rag_index') else '❌'}")
        st.write(f"Groq LLM: {'✅' if health.get('groq_llm') else '⚠️'}")
    else:
        st.error("Backend not reachable")

    st.markdown("---")
    languages = get_languages()
    language_name = st.selectbox("Report Language", list(languages.keys()), index=0)
    language_code = languages[language_name]

    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
- MRI Classification
- Grad-CAM Explainability
- Tumor Segmentation
- Clinical Report
- RAG Report
- Clinical Q&A
- Translation
""")

    st.markdown("---")
    st.caption("For educational use only. Not a final medical diagnosis.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <div class="title-main">🧠 X-Brain</div>
    <div class="title-sub">
        AI-powered brain tumor classification, segmentation, retrieval-grounded reporting, and clinical question answering.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Upload section ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Brain MRI Scan",
    type=["png", "jpg", "jpeg"],
    help="Upload a JPG or PNG MRI image."
)

if uploaded is None:
    st.markdown("""
    <div class="section-box">
        <div class="section-title">How it works</div>
        <div class="soft-text">
            1. Upload an MRI image<br>
            2. Run analysis<br>
            3. View classification, Grad-CAM, segmentation, clinical report, RAG report, and Q&A
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

pil_img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

preview_col, info_col = st.columns([1, 2])

with preview_col:
    st.image(pil_img, caption="Uploaded MRI", width=260)

with info_col:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scan Details</div>', unsafe_allow_html=True)
    st.write(f"**File:** `{uploaded.name}`")
    st.write(f"**Dimensions:** {pil_img.width} × {pil_img.height}")
    st.write(f"**Language:** {language_name}")
    run = st.button("🔬 Run Full Analysis", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

if run:
    with st.spinner("Running classification, Grad-CAM, segmentation, and RAG pipeline..."):
        try:
            result = call_analyze(uploaded, language_code)
            st.session_state.analysis_result = result
            st.session_state.analysis_file_name = uploaded.name
            st.session_state.analysis_language = language_code
            st.session_state.chat_history = []
            st.session_state.translated_report = None
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Run: uvicorn api.main:app --reload")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

if st.session_state.analysis_result is None:
    st.info("Click 'Run Full Analysis' to start.")
    st.stop()

if (
    st.session_state.analysis_file_name != uploaded.name
    or st.session_state.analysis_language != language_code
):
    st.warning("The selected file or language changed. Click 'Run Full Analysis' again.")
    st.stop()

result = st.session_state.analysis_result
clf = result["classification"]
seg = result["segmentation"]
report = result["clinical_report"]
rag = result["rag_report"]
imgs = result["images"]
ms = result["inference_time_ms"]

tumor_color = CLASS_COLORS.get(clf["class_name"], "#3b82f6")
urgency_color = URGENCY_COLORS.get(report.get("urgency", "NONE"), "#22c55e")

st.success(f"✅ Analysis complete in {ms} ms")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Summary",
    "🖼️ Visual Analysis",
    "📋 Clinical Report",
    "🤖 RAG Report",
    "💬 Q&A",
])

# ── Tab 1 Summary ─────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{tumor_color}">
            <div class="metric-label">Tumor Type</div>
            <div class="metric-value" style="color:{tumor_color}">
                {report.get("urgency_icon", "🧠")} {report.get("tumor_type", "Unknown")}
            </div>
            <div class="metric-sub">Predicted class</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#3b82f6">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{clf["confidence"]*100:.1f}%</div>
            <div class="metric-sub">{report.get("confidence_label", "—")}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        area_str = f'{seg["tumor_area_pct"]:.1f}%' if not seg["skipped"] else "N/A"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#f59e0b">
            <div class="metric-label">Tumor Area</div>
            <div class="metric-value">{area_str}</div>
            <div class="metric-sub">Scan region</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{urgency_color}">
            <div class="metric-label">Clinical Urgency</div>
            <div class="metric-value" style="color:{urgency_color}">
                {report.get("urgency", "—")}
            </div>
            <div class="metric-sub">Priority level</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classification Probabilities</div>', unsafe_allow_html=True)
    for cls, prob in sorted(clf["probabilities"].items(), key=lambda x: -x[1]):
        st.progress(float(prob), text=f"{cls.capitalize()} — {prob*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    if not seg["skipped"]:
        s1, s2, s3 = st.columns(3)
        s1.metric("Tumor Pixels", f'{seg["tumor_pixels"]:,}')
        s2.metric("Total Pixels", f'{seg["total_pixels"]:,}')
        s3.metric("Mask Present", "Yes" if seg["has_mask"] else "No")

# ── Tab 2 Visuals ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual Outputs</div>', unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        st.image(pil_img, caption="Original MRI", width=240)

    with v2:
        st.image(b64_to_bytes_img(imgs["gradcam_overlay"]), caption="Grad-CAM Overlay", width=240)

    with v3:
        st.image(b64_to_bytes_img(imgs["seg_overlay"]), caption="Segmentation Overlay", width=240)

    st.markdown('</div>', unsafe_allow_html=True)

    v4, v5 = st.columns(2)
    with v4:
        st.image(b64_to_bytes_img(imgs["gradcam_heatmap"]), caption="Grad-CAM Heatmap", width=280)
    with v5:
        st.image(b64_to_bytes_img(imgs["seg_mask"]), caption="Segmentation Mask", width=280)

# ── Tab 3 Clinical report ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.write(report.get("description", ""))
    st.caption(f'WHO Classification: {report.get("who_classification", "N/A")}')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Confidence Assessment</div>', unsafe_allow_html=True)
    st.info(f'{report.get("confidence_label", "—")} Confidence: {report.get("confidence_note", "")}')
    st.markdown('</div>', unsafe_allow_html=True)

    if report.get("symptoms"):
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Clinical Features & Symptoms</div>', unsafe_allow_html=True)
        for item in report["symptoms"]:
            st.markdown(f"- {item}")
        st.markdown('</div>', unsafe_allow_html=True)

    if report.get("diagnosis_methods"):
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Diagnostic Workup</div>', unsafe_allow_html=True)
        for item in report["diagnosis_methods"]:
            st.markdown(f"- {item}")
        st.markdown('</div>', unsafe_allow_html=True)

    if report.get("treatment_options"):
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Treatment Options</div>', unsafe_allow_html=True)
        for item in report["treatment_options"]:
            st.markdown(f"- {item}")
        st.markdown('</div>', unsafe_allow_html=True)

    if report.get("prognosis"):
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prognosis</div>', unsafe_allow_html=True)
        for k, v in report["prognosis"].items():
            st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer-box">{report.get("disclaimer", "")}</div>', unsafe_allow_html=True)

# ── Tab 4 RAG report ──────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">RAG-Enhanced Report</div>', unsafe_allow_html=True)
    st.markdown(rag.get("llm_report", "No RAG report available."))

    if rag.get("retrieved_docs"):
        st.markdown("### Retrieved Sources")
        chips = ""
        for src in rag["retrieved_docs"]:
            chips += f'<span class="source-chip">{src["source"]} · {src.get("score", 0):.2f}</span>'
        st.markdown(chips, unsafe_allow_html=True)

        with st.expander("View Retrieved Passages"):
            for i, doc in enumerate(rag["retrieved_docs"], 1):
                st.markdown(f"**[{i}] {doc['source']}**")
                st.write(doc.get("snippet", ""))
                st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Translate Report</div>', unsafe_allow_html=True)
    if st.button("🌐 Translate RAG Report"):
        try:
            st.session_state.translated_report = call_translate(
                rag.get("llm_report", ""),
                language_code,
            )
        except Exception as e:
            st.error(f"Translation error: {e}")

    if st.session_state.translated_report:
        st.write(st.session_state.translated_report)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 5 Q&A ────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Ask follow-up questions about this MRI analysis. Answers are based on the current report and retrieved medical literature.</div>', unsafe_allow_html=True)

    suggestions = {
        "glioma": [
            "What are the treatment options?",
            "What symptoms should be monitored?",
            "What does this confidence score mean?",
        ],
        "meningioma": [
            "Can this tumor be treated without surgery?",
            "What follow-up is usually needed?",
            "What symptoms are common?",
        ],
        "pituitary": [
            "Can this affect hormones?",
            "What symptoms are common?",
            "What is the usual treatment?",
        ],
        "notumor": [
            "What does no tumor detected mean?",
            "Should a follow-up MRI be done?",
            "Can symptoms still occur without a tumor?",
        ],
    }

    current_suggestions = suggestions.get(clf["class_name"], [])
    if current_suggestions:
        q1, q2, q3 = st.columns(3)
        cols = [q1, q2, q3]
        for idx, q in enumerate(current_suggestions):
            with cols[idx]:
                if st.button(q, key=f"suggest_{idx}"):
                    st.session_state["pending_question"] = q

    pending = st.session_state.pop("pending_question", "")
    question = st.text_input(
        "Ask a question",
        value=pending,
        placeholder="e.g. What are the treatment options?",
        key="qa_input",
    )

    c_ask, c_clear = st.columns([1, 1])
    with c_ask:
        ask = st.button("📨 Ask")
    with c_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask and question.strip():
        try:
            report_ctx = (
                f'Tumor: {clf["class_name"]} | '
                f'Confidence: {clf["confidence"]:.1%} | '
                f'Area: {seg["tumor_area_pct"]:.1f}% | '
                f'Urgency: {report.get("urgency", "")}\n\n'
                f'{rag.get("llm_report", "")[:1500]}'
            )
            answer = call_qa(
                question=question,
                report_ctx=report_ctx,
                class_name=clf["class_name"],
                history=st.session_state.chat_history,
                language=language_code,
            )

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error(f"Q&A error: {e}")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot"><b>X-Brain AI:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with st.expander("🔧 Raw API Response"):
    st.json(result)