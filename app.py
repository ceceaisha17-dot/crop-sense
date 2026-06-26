import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CROP-SENSE",
    page_icon="🌾",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #1a3d1f 0%, #2d6a35 60%, #4a8c52 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f5e9c8;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #b8d4bc;
    font-weight: 400;
}
.hero-tag {
    display: inline-block;
    background: rgba(245,233,200,0.15);
    color: #f5e9c8;
    border: 1px solid rgba(245,233,200,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

.result-box {
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-mentah {
    background: #e8f5e9;
    border: 2px solid #43a047;
}
.result-siap {
    background: #fffde7;
    border: 2px solid #f9a825;
}
.result-terlalu {
    background: #efebe9;
    border: 2px solid #795548;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-desc {
    font-size: 0.9rem;
    color: #555;
    margin-top: 0.4rem;
}
.metric-row {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.metric-chip {
    background: rgba(0,0,0,0.06);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #333;
}
.hue-bar-wrap {
    margin: 1.5rem 0 0.5rem 0;
}
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.model-agree {
    font-size: 0.85rem;
    color: #2e7d32;
    font-weight: 500;
    margin-top: 0.6rem;
}
.model-disagree {
    font-size: 0.85rem;
    color: #e65100;
    font-weight: 500;
    margin-top: 0.6rem;
}
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf      = joblib.load("rice_rf_model.pkl")
    dt      = joblib.load("rice_dt_model.pkl")
    scaler  = joblib.load("rice_scaler.pkl")
    le      = joblib.load("rice_label_encoder.pkl")
    return rf, dt, scaler, le

try:
    rf, dt, scaler, le = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(image_array):
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (128, 128))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(img_hsv)

    try:
        mask_s = s_ch > threshold_otsu(s_ch)
    except:
        mask_s = s_ch > 20
    try:
        mask_v = v_ch < threshold_otsu(v_ch)
    except:
        mask_v = v_ch < 50

    mask   = mask_s | mask_v
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask,                  cv2.MORPH_CLOSE, kernel)
    mask   = mask.astype(bool)
    if mask.sum() < 100:
        mask = np.ones(s_ch.shape, dtype=bool)

    h_mean    = float(h_ch[mask].mean())
    h_std     = float(np.std(h_ch[mask]))
    s_mean    = float(s_ch[mask].mean())
    s_std     = float(np.std(s_ch[mask]))
    v_mean    = float(v_ch[mask].mean())
    v_std     = float(np.std(v_ch[mask]))

    img_gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx    = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag       = np.sqrt(sobelx**2 + sobely**2)
    grad_mean = float(mag[mask].mean())
    grad_std  = float(mag[mask].std())

    return np.array([[h_mean, h_std, s_mean, s_std, v_mean, v_std, grad_mean, grad_std]]), \
           h_mean, s_mean, v_mean, grad_mean

# ── Hue gauge chart ───────────────────────────────────────────────────────────
def draw_hue_gauge(hue_value):
    fig, ax = plt.subplots(figsize=(7, 1.1))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # gradient bar: green → yellow → brown
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    colors_list = []
    for t in np.linspace(0, 1, 300):
        if t < 0.4:
            r = t / 0.4 * 0.3 + 0.1
            g = 0.55 + t / 0.4 * 0.1
            b = 0.1
        elif t < 0.7:
            tt = (t - 0.4) / 0.3
            r = 0.4 + tt * 0.55
            g = 0.65 - tt * 0.2
            b = 0.05
        else:
            tt = (t - 0.7) / 0.3
            r = 0.95 - tt * 0.45
            g = 0.45 - tt * 0.25
            b = 0.05 + tt * 0.15
        colors_list.append([r, g, b])

    bar_img = np.array([colors_list])
    ax.imshow(bar_img, aspect='auto', extent=[0, 70, 0, 1])

    # zones
    ax.axvspan(0,  22, alpha=0.0)
    ax.axvspan(22, 28, alpha=0.12, color='gray')  # transition
    ax.axvspan(28, 38, alpha=0.15, color='gold')  # optimal
    ax.axvspan(38, 70, alpha=0.0)

    # marker
    marker_x = min(max(hue_value, 0), 70)
    ax.axvline(marker_x, color='white', linewidth=2.5, zorder=5)
    ax.plot(marker_x, 0.5, 'o', color='white', markersize=9,
            markeredgecolor='#333', markeredgewidth=1.5, zorder=6)

    # zone labels
    ax.text(11,  -0.35, 'Terlalu\nMatang', ha='center', va='top', fontsize=7,
            color='#795548', fontweight='600')
    ax.text(33,  -0.35, 'Siap\nPanen', ha='center', va='top', fontsize=7,
            color='#f57f17', fontweight='600')
    ax.text(54,  -0.35, 'Mentah', ha='center', va='top', fontsize=7,
            color='#2e7d32', fontweight='600')

    ax.set_xlim(0, 70)
    ax.set_ylim(-0.5, 1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# ── Label config ──────────────────────────────────────────────────────────────
LABEL_CONFIG = {
    "Mentah": {
        "css":   "result-mentah",
        "emoji": "🌿",
        "color": "#2e7d32",
        "desc":  "Padi belum siap panen. Tunggu hingga warna bergeser ke kuning keemasan. Panen terlalu dini dapat meningkatkan persentase bulir hijau dan menurunkan kualitas beras giling."
    },
    "Siap_Panen": {
        "css":   "result-siap",
        "emoji": "🌾",
        "color": "#f57f17",
        "desc":  "Waktu panen optimal. Nilai Mean Hue berada pada rentang 28–38 yang mencerminkan warna kuning keemasan khas gabah siap tuai. Segera lakukan pemanenan."
    },
    "Terlalu_Matang": {
        "css":   "result-terlalu",
        "emoji": "🍂",
        "color": "#795548",
        "desc":  "Padi melewati masa panen optimal. Risiko shattering loss meningkat — gabah mudah rontok saat proses pemotongan. Prioritaskan pemanenan segera."
    }
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">SATRIA DATA 2026 · SEC</div>
    <div class="hero-title">🌾 CROP-SENSE</div>
    <div class="hero-sub">Color-based Rice Optimal Prediction System for Enhancing harvest Efficiency<br>
    Klasifikasi kematangan padi sawah berbasis fitur ruang warna HSV</div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"Gagal memuat model: {load_error}")
    st.info("Pastikan file `rice_rf_model.pkl`, `rice_dt_model.pkl`, `rice_scaler.pkl`, dan `rice_label_encoder.pkl` berada di folder yang sama dengan `app.py`.")
    st.stop()

# Upload
st.markdown('<div class="section-title">Upload Citra Padi</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Pilih gambar hamparan padi sawah",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Feature extraction & prediction
    with st.spinner("Mengekstraksi fitur HSV..."):
        feats, h_mean, s_mean, v_mean, grad_mean = extract_features(img_array)
        feats_scaled = scaler.transform(feats)

        pred_rf = le.inverse_transform([rf.predict(feats_scaled)[0]])[0]
        pred_dt = le.inverse_transform([dt.predict(feats)[0]])[0]

    with col2:
        cfg = LABEL_CONFIG[pred_rf]
        st.markdown(f"""
        <div class="result-box {cfg['css']}">
            <div style="font-size:2.2rem">{cfg['emoji']}</div>
            <div class="result-label" style="color:{cfg['color']}">{pred_rf.replace('_', ' ')}</div>
            <div class="metric-row">
                <span class="metric-chip">Hue {h_mean:.1f}</span>
                <span class="metric-chip">Sat {s_mean:.1f}</span>
                <span class="metric-chip">Val {v_mean:.1f}</span>
            </div>
            {"<div class='model-agree'>✓ Random Forest & Decision Tree sepakat</div>" 
             if pred_rf == pred_dt 
             else f"<div class='model-disagree'>⚠ Decision Tree: {pred_dt.replace('_',' ')}</div>"}
        </div>
        """, unsafe_allow_html=True)

    # Hue gauge
    st.markdown('<div class="section-title" style="margin-top:1.5rem">Posisi Mean Hue pada Spektrum Kematangan</div>', unsafe_allow_html=True)
    gauge_fig = draw_hue_gauge(h_mean)
    st.pyplot(gauge_fig, use_container_width=True)
    plt.close(gauge_fig)

    # Interpretation
    st.markdown(f"""
    <div style="background:#f9f9f9; border-radius:10px; padding:1rem 1.2rem; margin-top:0.5rem; font-size:0.88rem; color:#444; line-height:1.6;">
        {cfg['desc']}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#f5f5f5; border-radius:12px; padding:2rem; text-align:center; color:#999; margin-top:1rem;">
        <div style="font-size:2rem; margin-bottom:0.5rem">📷</div>
        <div style="font-weight:500">Upload foto hamparan padi sawah untuk memulai klasifikasi</div>
        <div style="font-size:0.82rem; margin-top:0.4rem">Mendukung format JPG, JPEG, PNG</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    CROP-SENSE · Satria Data 2026 SEC · Universitas Islam Indonesia<br>
    Random Forest (93.41%) + Decision Tree (92.31%) · Fitur: HSV + Gradien Sobel
</div>
""", unsafe_allow_html=True)
