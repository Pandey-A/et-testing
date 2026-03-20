import streamlit as st
import requests
import tempfile
import os
import time
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# Backend API URLs
# ─────────────────────────────────────────────
VIDEO_API_URL = "http://103.22.140.216:5006/predict/video"
IMAGE_API_URL = "http://103.22.140.216:5006/predict/image"
HEALTH_URL    = "http://103.22.140.216:5006/health"

# ─────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.mono { font-family: 'JetBrains Mono', monospace; }

.verdict-fake {
    background: linear-gradient(135deg, #ff4b4b22, #ff000011);
    border: 2px solid #ff4b4b;
    border-radius: 12px;
    padding: 22px 28px;
    text-align: center;
    margin-bottom: 12px;
}
.verdict-real {
    background: linear-gradient(135deg, #21c55d22, #00ff6611);
    border: 2px solid #21c55d;
    border-radius: 12px;
    padding: 22px 28px;
    text-align: center;
    margin-bottom: 12px;
}
.verdict-label {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 4px;
    margin: 0;
}
.fake-text { color: #ff4b4b; }
.real-text { color: #21c55d; }

.metric-box {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 0.70rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
}

.info-panel {
    background: #111827;
    border-left: 4px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.8;
}

/* Segment cards */
.seg-high  { background:#ff4b4b18; border:1px solid #ff4b4b66; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.seg-warn  { background:#f59e0b18; border:1px solid #f59e0b66; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.seg-ok    { background:#21c55d18; border:1px solid #21c55d66; border-radius:10px; padding:14px 18px; margin-bottom:10px; }

.seg-header { display:flex; align-items:center; gap:14px; margin-bottom:8px; }
.seg-time   { font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:700; color:#e0e0e0; }
.seg-badge  { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;
              padding:3px 10px; border-radius:20px; }
.badge-high { background:#ff4b4b33; color:#ff4b4b; border:1px solid #ff4b4b; }
.badge-warn { background:#f59e0b33; color:#f59e0b; border:1px solid #f59e0b; }
.badge-ok   { background:#21c55d33; color:#21c55d; border:1px solid #21c55d; }

.seg-stat-label { font-size:0.72rem; color:#aaa; text-transform:uppercase; letter-spacing:1px; }
.seg-stat-val   { font-family:'JetBrains Mono',monospace; font-size:1.15rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 DeepFake Detector")
    st.markdown("---")
    st.markdown("**Model info**")
    st.markdown(
        "- 🖼️ Image: mock-image-detector-v1\n"
        "- 🎬 Video: Segment-level lip & face analysis\n"
        "- 🔧 Backend: Flask"
    )
    st.markdown("---")
    st.markdown("**Server status**")
    try:
        hr = requests.get(HEALTH_URL, timeout=5)
        if hr.status_code == 200:
            hdata = hr.json()
            st.success(f"✅ {hdata.get('service', 'Server')} — online")
        else:
            st.error("❌ Server returned error")
    except Exception:
        st.error("❌ Cannot reach server")

    st.markdown("---")
    st.caption(f"Session started: {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    st.markdown("**Manipulation risk guide**")
    st.markdown("""
| % Range  | Risk Level     |
|----------|----------------|
| 0 – 15   | 🟢 Low         |
| 15 – 30  | 🟡 Moderate    |
| 30 – 50  | 🟠 High        |
| 50 +     | 🔴 Very High   |
""")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def risk_color(pct):
    if pct >= 50: return "#ff4b4b"
    if pct >= 30: return "#f97316"
    if pct >= 15: return "#f59e0b"
    return "#21c55d"

def risk_emoji(pct):
    if pct >= 50: return "🔴"
    if pct >= 30: return "🟠"
    if pct >= 15: return "🟡"
    return "🟢"

def risk_label(pct):
    if pct >= 50: return "Very High"
    if pct >= 30: return "High"
    if pct >= 15: return "Moderate"
    return "Low"

def seg_css(pct):
    if pct >= 30: return "seg-high", "badge-high", "HIGH RISK"
    if pct >= 15: return "seg-warn",  "badge-warn",  "MODERATE"
    return "seg-ok", "badge-ok", "LOW RISK"

def render_file_metadata(uploaded_file, media_type="image"):
    size_mb = uploaded_file.size / (1024 * 1024)
    ext = os.path.splitext(uploaded_file.name)[1].upper()
    st.markdown(f"""
    <div class="info-panel">
        <strong>📁 File metadata</strong><br>
        <span class="mono">Name :</span> {uploaded_file.name}<br>
        <span class="mono">Type :</span> {ext} ({media_type})<br>
        <span class="mono">Size :</span> {size_mb:.2f} MB ({uploaded_file.size:,} bytes)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VIDEO result renderer  (mock API format)
# Response shape:
#   overall_result, fake_clip_count, avg_lips, avg_face,
#   segments: [ { Time Range (s), lips Manipulation(%), Face Manipulation(%) } ]
# ─────────────────────────────────────────────
def render_video_result(result, elapsed):
    overall    = result.get("overall_result", "Unknown")
    fake_clips = result.get("fake_clip_count", 0)
    avg_lips   = float(result.get("avg_lips", 0.0))
    avg_face   = float(result.get("avg_face", 0.0))
    segments   = result.get("segments", [])

    is_fake = "fake" in overall.lower()
    css_cls = "verdict-fake" if is_fake else "verdict-real"
    txt_cls = "fake-text"    if is_fake else "real-text"
    icon    = "⚠️ DEEPFAKE DETECTED" if is_fake else "✅ AUTHENTIC CONTENT"

    # ── 1. Verdict banner ────────────────────
    st.markdown(f"""
    <div class="{css_cls}">
        <p class="verdict-label {txt_cls}">{icon}</p>
        <p style="margin:6px 0 0; color:#aaa; font-size:0.9rem;">
            Overall result: <strong>{overall}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"⏱️ Analysed in **{elapsed:.1f} s**")
    st.markdown("---")

    # ── 2. Top-level summary metrics ─────────
    st.markdown("### 📊 Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Overall Result</div>
            <div class="metric-value" style="font-size:1.05rem;
                 color:{'#ff4b4b' if is_fake else '#21c55d'};">{overall}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Fake Clip Count</div>
            <div class="metric-value">{fake_clips}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Avg Lips Manip.</div>
            <div class="metric-value" style="color:{risk_color(avg_lips)};">
                {avg_lips:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Avg Face Manip.</div>
            <div class="metric-value" style="color:{risk_color(avg_face)};">
                {avg_face:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. Average manipulation bars ─────────
    st.markdown("### 📈 Average Manipulation Levels")
    col_l, col_f = st.columns(2)
    with col_l:
        st.markdown(f"**👄 Lips Manipulation — {avg_lips:.2f}%** &nbsp; {risk_emoji(avg_lips)} {risk_label(avg_lips)}")
        st.progress(min(avg_lips / 100.0, 1.0))
    with col_f:
        st.markdown(f"**🧑 Face Manipulation — {avg_face:.2f}%** &nbsp; {risk_emoji(avg_face)} {risk_label(avg_face)}")
        st.progress(min(avg_face / 100.0, 1.0))

    st.markdown("---")

    # ── 4. Segment-by-segment cards ──────────
    if segments:
        st.markdown("### 🎞️ Segment-by-Segment Analysis")
        st.caption(f"{len(segments)} segments analysed · Each segment = 2 seconds of video")

        for idx, seg in enumerate(segments, 1):
            time_range = seg.get("Time Range (s)", "?")
            lips_pct   = float(seg.get("lips Manipulation(%)", 0.0))
            face_pct   = float(seg.get("Face Manipulation(%)", 0.0))
            worst      = max(lips_pct, face_pct)
            card_css, badge_css, badge_txt = seg_css(worst)

            st.markdown(f"""
            <div class="{card_css}">
                <div class="seg-header">
                    <span class="seg-time">Segment {idx} &nbsp;|&nbsp; ⏱ {time_range} s</span>
                    <span class="seg-badge {badge_css}">{badge_txt}</span>
                </div>
                <div style="display:flex; gap:40px; margin-top:4px;">
                    <div>
                        <div class="seg-stat-label">👄 Lips Manipulation</div>
                        <div class="seg-stat-val" style="color:{risk_color(lips_pct)};">{lips_pct:.1f}%</div>
                    </div>
                    <div>
                        <div class="seg-stat-label">🧑 Face Manipulation</div>
                        <div class="seg-stat-val" style="color:{risk_color(face_pct)};">{face_pct:.1f}%</div>
                    </div>
                    <div>
                        <div class="seg-stat-label">⚠️ Max Risk</div>
                        <div class="seg-stat-val" style="color:{risk_color(worst)};">{worst:.1f}%</div>
                    </div>
                    <div>
                        <div class="seg-stat-label">🔖 Risk Level</div>
                        <div class="seg-stat-val" style="color:{risk_color(worst)};">{risk_emoji(worst)} {risk_label(worst)}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bars per segment
            sb1, sb2 = st.columns(2)
            with sb1:
                st.caption(f"👄 Lips — {lips_pct:.1f}%")
                st.progress(min(lips_pct / 100.0, 1.0))
            with sb2:
                st.caption(f"🧑 Face — {face_pct:.1f}%")
                st.progress(min(face_pct / 100.0, 1.0))

        st.markdown("---")

        # ── 5. Summary data table ─────────────
        st.markdown("### 📋 Segment Data Table")
        rows = []
        for idx, seg in enumerate(segments, 1):
            lips_pct = float(seg.get("lips Manipulation(%)", 0.0))
            face_pct = float(seg.get("Face Manipulation(%)", 0.0))
            worst    = max(lips_pct, face_pct)
            rows.append({
                "Seg #":               idx,
                "Time Range (s)":      seg.get("Time Range (s)", "?"),
                "Lips Manip. (%)":     round(lips_pct, 2),
                "Face Manip. (%)":     round(face_pct, 2),
                "Max Manip. (%)":      round(worst, 2),
                "Risk Level":          f"{risk_emoji(worst)} {risk_label(worst)}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── 6. Chart ──────────────────────────
        st.markdown("### 📉 Manipulation Over Time")
        chart_df = pd.DataFrame({
            "Time Segment": [s.get("Time Range (s)", str(i)) for i, s in enumerate(segments)],
            "Lips Manipulation (%)": [float(s.get("lips Manipulation(%)", 0)) for s in segments],
            "Face Manipulation (%)": [float(s.get("Face Manipulation(%)", 0)) for s in segments],
        }).set_index("Time Segment")
        st.line_chart(chart_df, use_container_width=True)

    st.markdown("---")

    # ── 7. Interpretation panel ───────────────
    colour = "#ff4b4b" if is_fake else "#21c55d"
    if is_fake:
        interp = (
            f"The video was classified as <strong>{overall}</strong>. "
            f"<strong>{fake_clips}</strong> segment(s) showed manipulated content. "
            f"Average lip manipulation across all segments: <strong>{avg_lips:.2f}%</strong> "
            f"· Average face manipulation: <strong>{avg_face:.2f}%</strong>. "
            "Segments highlighted in red/orange warrant the closest scrutiny."
        )
    else:
        interp = (
            f"The video was classified as <strong>{overall}</strong>. "
            "No significant manipulation was detected across any segment. "
            f"Average lip manipulation: <strong>{avg_lips:.2f}%</strong> "
            f"· Average face manipulation: <strong>{avg_face:.2f}%</strong>."
        )

    st.markdown(f"""
    <div class="info-panel" style="border-left-color:{colour};">
        <strong>📊 Interpretation</strong><br><br>{interp}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛠️ Raw API response"):
        st.json(result)


# ─────────────────────────────────────────────
# IMAGE result renderer  (mock API format)
# Response shape: { prediction, confidence, model }
# ─────────────────────────────────────────────
def render_image_result(result, elapsed, img_path):
    prediction = result.get("prediction", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    model_name = result.get("model", "unknown")
    conf_pct   = confidence * 100
    is_fake    = prediction.lower() == "fake"

    css_cls = "verdict-fake" if is_fake else "verdict-real"
    txt_cls = "fake-text"    if is_fake else "real-text"
    icon    = "⚠️ DEEPFAKE DETECTED" if is_fake else "✅ AUTHENTIC CONTENT"

    # ── 1. Verdict banner ────────────────────
    st.markdown(f"""
    <div class="{css_cls}">
        <p class="verdict-label {txt_cls}">{icon}</p>
        <p style="margin:6px 0 0; color:#aaa; font-size:0.9rem;">
            Model: <strong>{model_name}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"⏱️ Analysed in **{elapsed:.2f} s**")
    st.markdown("---")

    # ── 2. Summary metrics ───────────────────
    st.markdown("### 📊 Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Prediction</div>
            <div class="metric-value" style="color:{'#ff4b4b' if is_fake else '#21c55d'};">
                {prediction}
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{conf_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        rl = risk_label(conf_pct)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value" style="font-size:1rem; padding-top:6px;
                 color:{risk_color(conf_pct)};">{risk_emoji(conf_pct)} {rl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. Confidence bars ───────────────────
    st.markdown("### 📈 Confidence Breakdown")
    col_a, col_b = st.columns(2)
    with col_a:
        label_a = "🤖 Fake confidence" if is_fake else "👤 Real confidence"
        st.markdown(f"**{label_a} — {conf_pct:.1f}%**")
        st.progress(float(confidence))
    with col_b:
        opp      = 1.0 - confidence
        label_b  = "👤 Real confidence" if is_fake else "🤖 Fake confidence"
        st.markdown(f"**{label_b} — {opp*100:.1f}%**")
        st.progress(float(opp))

    st.markdown("---")

    # ── 4. Image + detailed score table ──────
    st.markdown("### 🖼️ Image Analysis Detail")
    ia, ib = st.columns([1, 2])
    with ia:
        st.image(img_path, caption="Analysed image", use_column_width=True)
    with ib:
        st.markdown("**Full score breakdown**")
        st.markdown(f"""
| Metric | Value |
|---|---|
| Prediction | **{prediction}** |
| Confidence | **{conf_pct:.2f}%** |
| Raw confidence score | `{confidence:.6f}` |
| Opposite class score | `{1 - confidence:.6f}` |
| Model used | `{model_name}` |
| Risk level | {risk_emoji(conf_pct)} {risk_label(conf_pct)} |
""")

    st.markdown("---")

    # ── 5. Interpretation ────────────────────
    colour = "#ff4b4b" if is_fake else "#21c55d"
    if is_fake:
        interp = (
            f"The model <strong>{model_name}</strong> classified this image as "
            f"<strong>Fake</strong> with <strong>{conf_pct:.1f}% confidence</strong>. "
            "The image may have been generated or manipulated using AI. "
            "Verify the source before using or sharing it."
        )
    else:
        interp = (
            f"The model <strong>{model_name}</strong> classified this image as "
            f"<strong>Real</strong> with <strong>{conf_pct:.1f}% confidence</strong>. "
            "No significant signs of AI manipulation were detected. "
            "Confidence below 70% still warrants additional scrutiny."
        )

    st.markdown(f"""
    <div class="info-panel" style="border-left-color:{colour};">
        <strong>📊 Interpretation</strong><br><br>{interp}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛠️ Raw API response"):
        st.json(result)


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.title("🔍 DeepFake Detection System")
st.markdown("Upload a **video** or **image** to analyse it for AI-generated or manipulated content.")
st.markdown("---")

tab_video, tab_image = st.tabs(["🎬  Video Analysis", "🖼️  Image Analysis"])


# ══════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════
with tab_video:
    st.header("Video Deepfake Detection")
    st.markdown("Supported formats: `mp4 · avi · mov · mkv`  •  Max size: **100 MB**")

    uploaded_video = st.file_uploader(
        "Drop a video file here or click to browse",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video:
        render_file_metadata(uploaded_video, "video")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.video(tmp_path)
        st.markdown("")

        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
            st.markdown("---")
            start_time = time.time()

            with open(tmp_path, "rb") as f:
                files = {"file": (uploaded_video.name, f, "video/mp4")}
                with st.spinner("🔄 Extracting segments and running inference…"):
                    try:
                        response = requests.post(VIDEO_API_URL, files=files, timeout=300)
                        elapsed  = time.time() - start_time

                        if response.status_code == 200:
                            render_video_result(response.json(), elapsed)
                        else:
                            st.error(f"❌ Server returned HTTP {response.status_code}")
                            st.json(response.json())

                    except requests.exceptions.Timeout:
                        st.error("⏰ Request timed out. Try a shorter/smaller video.")
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Cannot connect to server. Check status in the sidebar.")
                    except Exception as ex:
                        st.error(f"Unexpected error: {ex}")

        os.remove(tmp_path)


# ══════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════
with tab_image:
    st.header("Image Deepfake Detection")
    st.markdown("Supported formats: `jpg · jpeg · png · bmp`  •  Recommended: **face-forward portrait**")

    uploaded_image = st.file_uploader(
        "Drop an image file here or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        key="image_uploader",
    )

    if uploaded_image:
        render_file_metadata(uploaded_image, "image")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_image.read())
            tmp_path = tmp.name

        col_img, col_ctrl = st.columns([1, 1])
        with col_img:
            st.image(tmp_path, caption="Uploaded image", use_column_width=True)
        with col_ctrl:
            st.markdown("### Ready to analyse")
            st.markdown(
                "The **mock-image-detector-v1** model will classify this image as "
                "**REAL** or **FAKE** with a full confidence breakdown."
            )
            run_btn = st.button("🚀 Analyze Image", type="primary", use_container_width=True)

        if run_btn:
            st.markdown("---")
            start_time = time.time()

            with open(tmp_path, "rb") as f:
                files = {"file": (uploaded_image.name, f, "image/png")}
                with st.spinner("🔄 Running inference…"):
                    try:
                        response = requests.post(IMAGE_API_URL, files=files, timeout=60)
                        elapsed  = time.time() - start_time

                        if response.status_code == 200:
                            render_image_result(response.json(), elapsed, tmp_path)
                        else:
                            st.error(f"❌ Server returned HTTP {response.status_code}")
                            st.json(response.json())

                    except requests.exceptions.Timeout:
                        st.error("⏰ Request timed out. Try a smaller image.")
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Cannot connect to server. Check status in the sidebar.")
                    except Exception as ex:
                        st.error(f"Unexpected error: {ex}")

        os.remove(tmp_path)
