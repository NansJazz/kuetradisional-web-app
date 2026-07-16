import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import json
from PIL import Image
import os
import base64

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Klasifikasi Kue Tradisional Indonesia",
    page_icon="🍰",
    layout="wide"
)

MODEL_TEST_ACCURACY = 90.0
CONFIDENCE_THRESHOLD = 70.0

# Jika model kamu sudah memakai preprocess_input saat training di dalam model,
# biarkan False. Jika model kamu belum memakai preprocess_input di dalam model,
# ubah menjadi True.
USE_EXTERNAL_PREPROCESS = False

# =========================
# FUNGSI GAMBAR DAN BACKGROUND
# =========================
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def get_mime_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        return "image/png"
    elif ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    return "image/png"

def render_hero(title, subtitle, image_path, badge="Sistem Klasifikasi Citra"):
    image_base64 = get_base64_image(image_path)

    if image_base64:
        mime_type = get_mime_type(image_path)
        background_style = f"""
        background-image:
        linear-gradient(90deg, rgba(43,27,18,0.90), rgba(58,36,23,0.78), rgba(58,36,23,0.32)),
        url('data:{mime_type};base64,{image_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        """
    else:
        background_style = """
        background: linear-gradient(135deg, #2B1B12, #3A2417, #7C4315);
        """

    hero_html = f"""
<div class="hero" style="{background_style}">
    <div class="hero-badge">{badge}</div>
    <div class="hero-title">{title}</div>
    <div class="hero-subtitle">{subtitle}</div>
    <div class="hero-stats">
        <div class="hero-stat-item">
            <div class="hero-stat-value">MobileNetV2</div>
            <div class="hero-stat-label">Arsitektur Model</div>
        </div>
        <div class="hero-stat-item">
            <div class="hero-stat-value">6 Kelas</div>
            <div class="hero-stat-label">Jenis Kue</div>
        </div>
        <div class="hero-stat-item">
            <div class="hero-stat-value">{MODEL_TEST_ACCURACY:.0f}%</div>
            <div class="hero-stat-label">Akurasi Model</div>
        </div>
    </div>
</div>
"""

    st.markdown(hero_html, unsafe_allow_html=True)

def render_section(title, subtitle):
    st.markdown(
        f"""
        <div class="section-wrapper">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_confidence_status(confidence):
    if confidence >= 80:
        return "Tinggi", "Model memiliki keyakinan tinggi terhadap hasil prediksi.", "success-box"
    elif confidence >= 60:
        return "Sedang", "Model cukup yakin, tetapi gambar masih perlu diperhatikan.", "warning-box"
    else:
        return "Rendah", "Model kurang yakin. Gambar mungkin kurang jelas atau mirip dengan kelas lain.", "error-box"

# =========================
# CSS MODERN
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,900&family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --ink: #2B1B12;
    --ink-soft: #4A3626;
    --text-muted: #6B5645;
    --bg-base: #FBF3E6;
    --bg-card: #FFFDF8;
    --border-warm: #E8D9C3;
    --caramel: #B5651D;
    --caramel-dark: #7C4315;
    --caramel-light: #F3E1C8;
    --pandan: #3F6B4A;
    --pandan-dark: #2C4E36;
    --pandan-light: #E1EBE0;
    --gold: #D9A441;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink-soft);
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(217, 164, 65, 0.16), transparent 38%),
        radial-gradient(circle at 96% 8%, rgba(63, 107, 74, 0.12), transparent 34%),
        radial-gradient(circle at 50% 100%, rgba(181, 101, 29, 0.08), transparent 45%),
        var(--bg-base);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

/* NAVIGASI */
.nav-container {
    background: rgba(255, 253, 248, 0.88);
    backdrop-filter: blur(16px);
    padding: 14px 18px;
    margin-bottom: 24px;
    border-radius: 20px;
    border: 1px solid var(--border-warm);
    box-shadow: 0 12px 30px rgba(74, 54, 38, 0.10);
}

div[role="radiogroup"] {
    gap: 18px;
}

div[role="radiogroup"] label {
    background: transparent !important;
    padding: 8px 10px;
    border-radius: 14px;
}

div[role="radiogroup"] label p {
    color: var(--ink) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

div[role="radiogroup"] label:has(input:checked) p {
    color: var(--caramel-dark) !important;
    font-weight: 900 !important;
}

div[role="radiogroup"] input[type="radio"] {
    accent-color: var(--caramel);
}

/* HERO */
.hero {
    position: relative;
    padding: 78px 64px;
    border-radius: 32px;
    color: white;
    margin-bottom: 54px;
    min-height: 430px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 24px 60px rgba(43, 27, 18, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.14);
    overflow: hidden;
}

/* Pinggiran scallop, terinspirasi lapisan kue lapis */
.hero::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -19px;
    height: 38px;
    background: radial-gradient(circle at 19px 0, transparent 19px, var(--gold) 20px) 0 0/38px 38px;
    pointer-events: none;
}

.hero * {
    color: white !important;
}

.hero-badge {
    width: fit-content;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(217, 164, 65, 0.55);
    padding: 9px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 22px;
    letter-spacing: 0.3px;
}

.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 56px;
    font-weight: 700;
    max-width: 850px;
    line-height: 1.08;
    margin-bottom: 18px;
    font-optical-sizing: auto;
}

.hero-subtitle {
    font-size: 19px;
    max-width: 880px;
    line-height: 1.75;
    color: #F3E9DA !important;
}

.hero-stats {
    display: flex;
    gap: 14px;
    margin-top: 34px;
    flex-wrap: wrap;
}

.hero-stat-item {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(217, 164, 65, 0.35);
    border-radius: 20px;
    padding: 16px 22px;
    min-width: 150px;
    backdrop-filter: blur(12px);
}

.hero-stat-value {
    font-family: 'Fraunces', serif;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 4px;
}

.hero-stat-label {
    font-size: 13px;
    font-weight: 600;
    color: #E4D6C2 !important;
}

/* SECTION */
.section-wrapper {
    margin-top: 8px;
    margin-bottom: 22px;
}

.section-title {
    font-family: 'Fraunces', serif;
    font-size: 34px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 8px;
    letter-spacing: -0.3px;
}

.section-subtitle {
    font-size: 16px;
    color: var(--text-muted);
    line-height: 1.75;
    max-width: 980px;
}

/* CARD UMUM */
.card {
    background: var(--bg-card);
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 14px 36px rgba(74, 54, 38, 0.08);
    border: 1px solid var(--border-warm);
    margin-bottom: 18px;
    min-height: 230px;
}

.card:hover {
    transform: translateY(-3px);
    transition: 0.25s ease;
    box-shadow: 0 20px 46px rgba(74, 54, 38, 0.14);
}

.card-icon {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: linear-gradient(135deg, var(--caramel-light), #ECD3A8);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 16px;
}

.card-title {
    font-family: 'Fraunces', serif;
    font-size: 21px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 9px;
}

.card-origin {
    font-size: 14px;
    font-weight: 800;
    color: var(--pandan-dark);
    margin-bottom: 12px;
}

.card-text {
    font-size: 15px;
    color: var(--text-muted);
    line-height: 1.7;
}

/* ABOUT */
.about-card {
    background: var(--bg-card);
    border-radius: 26px;
    padding: 30px;
    box-shadow: 0 14px 36px rgba(74, 54, 38, 0.08);
    border: 1px solid var(--border-warm);
    margin-bottom: 20px;
}

.about-title {
    font-family: 'Fraunces', serif;
    font-size: 23px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 12px;
}

.about-text {
    font-size: 16px;
    color: var(--ink-soft);
    line-height: 1.85;
    text-align: justify;
}

.step-card {
    background: linear-gradient(135deg, var(--bg-card), #FBF3E6);
    border: 1px solid var(--border-warm);
    border-radius: 22px;
    padding: 24px;
    min-height: 190px;
    box-shadow: 0 10px 26px rgba(74, 54, 38, 0.07);
}

.step-number {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: var(--pandan);
    color: white;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
}

.step-title {
    font-family: 'Fraunces', serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 8px;
}

.step-text {
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1.65;
}

/* KLASIFIKASI */
.classification-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-warm);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 16px 42px rgba(74, 54, 38, 0.09);
    margin-bottom: 20px;
}

.upload-title {
    font-family: 'Fraunces', serif;
    font-size: 25px;
    font-weight: 700;
    color: var(--ink) !important;
    margin-bottom: 8px;
}

.upload-desc {
    font-size: 15px;
    color: var(--text-muted) !important;
    line-height: 1.65;
    margin-bottom: 18px;
}

.result-title {
    font-family: 'Fraunces', serif;
    font-size: 23px;
    font-weight: 700;
    color: var(--ink) !important;
    margin-bottom: 12px;
}

.result-name {
    font-family: 'Fraunces', serif;
    font-size: 40px;
    font-weight: 700;
    color: var(--caramel-dark) !important;
    line-height: 1.12;
    margin-bottom: 12px;
}

.info-text {
    font-size: 16px;
    color: var(--ink-soft) !important;
    line-height: 1.85;
}

.metric-card {
    background: linear-gradient(135deg, var(--pandan-light), #D3E3D1);
    border: 1px solid #BFD8BC;
    border-radius: 20px;
    padding: 18px;
    margin-bottom: 12px;
}

.metric-label {
    font-size: 13px;
    font-weight: 900;
    color: var(--pandan-dark) !important;
    margin-bottom: 7px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.metric-value {
    font-family: 'Fraunces', serif;
    font-size: 31px;
    font-weight: 700;
    color: var(--ink) !important;
}

.success-box {
    background-color: #E4F1E2;
    border: 1px solid var(--pandan);
    color: var(--pandan-dark) !important;
    padding: 15px 17px;
    border-radius: 16px;
    font-weight: 800;
    margin-top: 12px;
    line-height: 1.55;
}

.warning-box {
    background-color: #FBEBCB;
    border: 1px solid var(--gold);
    color: #7A5A15 !important;
    padding: 15px 17px;
    border-radius: 16px;
    font-weight: 800;
    margin-top: 12px;
    line-height: 1.55;
}

.error-box {
    background-color: #F7E0DA;
    border: 1px solid #C1512F;
    color: #8A3719 !important;
    padding: 15px 17px;
    border-radius: 16px;
    font-weight: 800;
    margin-top: 12px;
    line-height: 1.55;
}

.prediction-item {
    background: var(--bg-base);
    border: 1px solid var(--border-warm);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 12px;
}

.prediction-name {
    font-size: 16px;
    font-weight: 900;
    color: var(--ink) !important;
    margin-bottom: 5px;
}

.prediction-score {
    font-size: 15px;
    font-weight: 800;
    color: var(--caramel-dark) !important;
}

.note-box {
    background: var(--bg-base);
    border-left: 5px solid var(--caramel);
    border-radius: 18px;
    padding: 18px 20px;
    color: var(--ink-soft) !important;
    line-height: 1.7;
    font-size: 15px;
}

/* Streamlit bawaan agar font tidak samar */
div[data-testid="stFileUploader"] {
    color: var(--ink) !important;
}

div[data-testid="stFileUploader"] * {
    color: var(--ink) !important;
}

div[data-testid="stFileUploader"] button {
    background: var(--caramel) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 800 !important;
}

div[data-testid="stFileUploader"] section {
    background: var(--bg-base) !important;
    border: 2px dashed #D8C29A !important;
    border-radius: 20px !important;
}

div[data-testid="stAlert"] p {
    color: var(--ink-soft) !important;
    font-weight: 600 !important;
}

/* Hilangkan background putih bawaan Streamlit */

div[data-testid="stVerticalBlockBorderWrapper"]{
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    border-radius: 0 !important;
}

div[data-testid="stVerticalBlock"]{
    background: transparent !important;
}

div[data-testid="stElementContainer"]{
    background: transparent !important;
}
            
section.main > div {
    background: transparent !important;
}

div.block-container {
    background: transparent !important;
}

main {
    background: transparent !important;
}

.stMarkdown, .stText, p, span, label {
    color: var(--ink-soft);
}

/* TOMBOL */
div.stButton > button {
    background: linear-gradient(135deg, var(--caramel), var(--caramel-dark));
    color: white;
    border: none;
    border-radius: 16px;
    padding: 13px 25px;
    font-weight: 900;
    font-size: 16px;
    margin-bottom: 25px;
    box-shadow: 0 12px 28px rgba(181, 101, 29, 0.30);
}

div.stButton > button:hover {
    background: linear-gradient(135deg, var(--caramel-dark), #5E3410);
    color: white;
    transform: translateY(-2px);
}

/* FOOTER */
.footer {
    background: linear-gradient(135deg, #2B1B12, #3A2417);
    color: #D8C7B0;
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    margin-top: 45px;
    font-size: 15px;
    line-height: 1.8;
    box-shadow: 0 18px 42px rgba(43, 27, 18, 0.22);
}

.footer b {
    font-family: 'Fraunces', serif;
    color: white;
}

/* RESPONSIVE */
@media only screen and (max-width: 768px) {
    .hero {
        padding: 50px 28px;
        min-height: 390px;
    }

    .hero-title {
        font-size: 36px;
    }

    .hero-subtitle {
        font-size: 16px;
    }

    .hero-stats {
        flex-direction: column;
    }

    .section-title {
        font-size: 28px;
    }

    .result-name {
        font-size: 32px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL DAN LABEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_kue.keras")

try:
    model = load_model()
except Exception as e:
    st.error("Model gagal dimuat. Pastikan file model_kue.keras berada satu folder dengan app.py.")
    st.stop()

try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except Exception as e:
    st.error("File class_names.json gagal dibaca. Pastikan file tSersebut berada satu folder dengan app.py.")
    st.stop()

# =========================
# DATA INFORMASI KUE
# =========================
kue_info = {
    "kue_dadar_gulung": {
        "nama": "Kue Dadar Gulung",
        "asal": "Jajanan tradisional Indonesia yang banyak dikenal di daerah Jawa.",
        "deskripsi": "Dadar gulung adalah kue tradisional berbentuk gulungan yang umumnya berwarna hijau dari daun pandan. Bagian dalamnya berisi parutan kelapa yang dimasak dengan gula merah sehingga memiliki rasa manis dan gurih.",
        "ciri": "Berbentuk gulungan, berwarna hijau, dan memiliki isian kelapa gula merah.",
        "ikon": "🌯"
    },
    "kue_klepon": {
        "nama": "Kue Klepon",
        "asal": "Jajanan tradisional Jawa yang populer di berbagai daerah Indonesia.",
        "deskripsi": "Klepon adalah kue berbentuk bulat kecil yang terbuat dari tepung ketan. Bagian dalamnya berisi gula merah cair, sedangkan bagian luarnya dilapisi parutan kelapa.",
        "ciri": "Bulat kecil, berwarna hijau, dan bagian luar dilapisi kelapa parut.",
        "ikon": "🟢"
    },
    "kue_lapis": {
        "nama": "Kue Lapis",
        "asal": "Kue tradisional Indonesia yang banyak ditemukan di berbagai daerah.",
        "deskripsi": "Kue lapis memiliki ciri khas berupa susunan warna berlapis. Kue ini biasanya dibuat dari tepung beras, tepung tapioka, santan, dan gula, lalu dikukus secara bertahap.",
        "ciri": "Memiliki warna berlapis dan tekstur kenyal.",
        "ikon": "🍰"
    },
    "kue_lumpur": {
        "nama": "Kue Lumpur",
        "asal": "Jajanan tradisional Indonesia yang populer di berbagai daerah.",
        "deskripsi": "Kue lumpur memiliki tekstur lembut dan padat dengan rasa manis gurih. Bahan yang umum digunakan antara lain kentang, tepung, telur, santan, dan gula.",
        "ciri": "Bentuk bulat pipih, tekstur lembut, dan biasanya berwarna kuning kecokelatan.",
        "ikon": "🧁"
    },
    "kue_risoles": {
        "nama": "Kue Risoles",
        "asal": "Makanan ringan yang populer di Indonesia dan mendapat pengaruh dari kuliner Eropa.",
        "deskripsi": "Risoles adalah makanan ringan berbentuk gulungan atau lipatan kulit tipis yang berisi sayuran, daging, atau ragout, kemudian dilapisi tepung panir dan digoreng.",
        "ciri": "Berwarna cokelat keemasan, dilapisi tepung panir, dan berbentuk gulungan atau segitiga.",
        "ikon": "🥟"
    },
    "kue_serabi": {
        "nama": "Kue Serabi",
        "asal": "Jajanan tradisional yang banyak dikenal di daerah Jawa dan Sunda.",
        "deskripsi": "Serabi adalah kue tradisional berbahan dasar tepung beras dan santan. Kue ini biasanya dimasak menggunakan cetakan kecil sehingga menghasilkan aroma khas dan tekstur lembut.",
        "ciri": "Berbentuk bulat, bagian tengah lembut, dan sering memiliki pinggiran agak kering.",
        "ikon": "🥞"
    }
}

def get_info_kue(class_key):
    return kue_info.get(class_key, {
        "nama": class_key.replace("_", " ").title(),
        "asal": "Informasi asal belum tersedia.",
        "deskripsi": "Deskripsi belum tersedia.",
        "ciri": "Ciri visual belum tersedia.",
        "ikon": "🍽️"
    })

# =========================
# SESSION STATE MENU
# =========================
menu_options = ["Beranda", "Tentang", "Nama Makanan", "Klasifikasi", "Kontak"]

if "menu_radio" not in st.session_state:
    st.session_state.menu_radio = "Beranda"

def pindah_ke_klasifikasi():
    st.session_state.menu_radio = "Klasifikasi"

# =========================
# NAVIGASI
# =========================
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

menu = st.radio(
    "",
    menu_options,
    horizontal=True,
    key="menu_radio"
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HALAMAN BERANDA
# =========================
if menu == "Beranda":
    render_hero(
        "Dari Gambar Menjadi Nama Kue Tradisional Indonesia",
        "Sistem ini membantu pengguna mengenali jenis kue tradisional Indonesia melalui gambar. Model MobileNetV2 digunakan untuk melakukan klasifikasi citra dan menampilkan informasi kue secara otomatis.",
        "assets/bg_beranda.jpg",
        "Klasifikasi Kue Tradisional Indonesia"
    )

    st.button(
        "Mulai Klasifikasi Sekarang",
        type="primary",
        on_click=pindah_ke_klasifikasi
    )

    render_section(
        "Fitur Utama Sistem",
        "Aplikasi ini dibuat untuk mempermudah proses pengenalan kue tradisional Indonesia secara otomatis melalui gambar yang diunggah oleh pengguna."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Import Gambar</div>
            <div class="card-text">
                Pengguna dapat mengunggah gambar kue dalam format JPG, JPEG, atau PNG.
                Gambar yang diunggah akan diproses terlebih dahulu sebelum masuk ke model klasifikasi.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Prediksi Otomatis</div>
            <div class="card-text">
                Sistem langsung memprediksi nama kue berdasarkan gambar yang diinput.
                Pengguna tidak perlu memilih label atau menebak nama makanan secara manual.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Informasi Hasil</div>
            <div class="card-text">
                Sistem menampilkan nama kue, confidence prediksi, akurasi model,
                top 3 prediksi, grafik probabilitas, serta penjelasan singkat mengenai kue.
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_section(
        "Alur Penggunaan Aplikasi",
        "Alur dibuat sederhana agar pengguna umum tetap mudah memahami cara kerja sistem."
    )

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-title">Buka Halaman Klasifikasi</div>
            <div class="step-text">Pengguna masuk ke menu klasifikasi untuk memulai proses pengenalan gambar kue.</div>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-title">Upload Gambar</div>
            <div class="step-text">Pengguna memilih gambar kue tradisional dari perangkat.</div>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-title">Model Memproses</div>
            <div class="step-text">Gambar diproses menjadi ukuran 224 x 224 piksel lalu dianalisis oleh model.</div>
        </div>
        """, unsafe_allow_html=True)

    with step4:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">4</div>
            <div class="step-title">Hasil Ditampilkan</div>
            <div class="step-text">Sistem menampilkan nama kue, confidence, dan informasi tambahan secara otomatis.</div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# HALAMAN TENTANG
# =========================
elif menu == "Tentang":
    render_hero(
        "Tentang Sistem Klasifikasi Kue Tradisional",
        "Sistem ini dibuat sebagai aplikasi berbasis Streamlit untuk mengenali kue tradisional Indonesia menggunakan citra dan model deep learning MobileNetV2.",
        "assets/bg_beranda.jpg",
        "Tentang Penelitian"
    )

# =========================
# HALAMAN NAMA MAKANAN
# =========================
elif menu == "Nama Makanan":
    render_hero(
        "Nama-Nama Kue Tradisional",
        "Halaman ini menampilkan daftar kue tradisional Indonesia yang dapat dikenali oleh sistem klasifikasi beserta ciri visual dan deskripsi singkatnya.",
        "assets/bg_nama_makanan.jpg",
        "Daftar Kelas Kue"
    )

    render_section(
        "Kue yang Dapat Diklasifikasikan",
        "Sistem ini berfokus pada enam jenis kue tradisional Indonesia sesuai batasan penelitian."
    )

    keys = list(kue_info.keys())

    for i in range(0, len(keys), 3):
        cols = st.columns(3)

        for idx, key in enumerate(keys[i:i+3]):
            info = get_info_kue(key)

            with cols[idx]:
                st.markdown(f"""
                <div class="card">
                    <div class="card-icon">{info["ikon"]}</div>
                    <div class="card-title">{info["nama"]}</div>
                    <div class="card-origin">{info["asal"]}</div>
                    <div class="card-text">
                        <b>Ciri Visual:</b> {info["ciri"]}<br><br>
                        {info["deskripsi"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =========================
# HALAMAN KLASIFIKASI
# =========================
elif menu == "Klasifikasi":
    render_hero(
        "Klasifikasi Gambar Kue",
        "Unggah gambar kue tradisional, kemudian sistem akan memprediksi nama kue secara otomatis menggunakan model MobileNetV2.",
        "assets/bg_beranda.jpg",
        "Prediksi Otomatis"
    )

    render_section(
        "Uji Klasifikasi Gambar",
        "Pada halaman ini pengguna hanya perlu mengunggah gambar. Sistem akan langsung menampilkan hasil prediksi tanpa perlu memilih label makanan secara manual."
    )

    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        st.markdown('<div class="classification-panel">', unsafe_allow_html=True)
        st.markdown('<div class="upload-title">Import Gambar Kue</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="upload-desc">Pilih gambar kue tradisional dalam format JPG, JPEG, atau PNG. Setelah gambar diunggah, sistem akan langsung melakukan prediksi otomatis.</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            label="Pilih gambar kue tradisional",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Klik Browse files untuk memilih gambar dari perangkat."
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)
        else:
            st.info("Silahkan unggah gambar kue terlebih dahulu untuk melakukan klasifikasi.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="note-box">
            <b>Catatan:</b> Gunakan gambar yang jelas, tidak terlalu gelap, dan objek kue terlihat dominan.
            Kualitas gambar dapat memengaruhi tingkat keyakinan model.
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        if uploaded_file is not None:
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized).astype(np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            if USE_EXTERNAL_PREPROCESS:
                img_array = preprocess_input(img_array)

            prediction = model.predict(img_array, verbose=0)

            predicted_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)

            # Jika confidence terlalu rendah
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = None
                info = None
            else:
                predicted_class = class_names[predicted_index]
                info = get_info_kue(predicted_class)

            status_label, status_text, status_class = get_confidence_status(confidence)

            st.markdown('<div class="classification-panel">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Hasil Prediksi Otomatis</div>', unsafe_allow_html=True)

            if info is None:
                st.markdown(
                    '<div class="result-name" style="color:#C0392B !important;">Gambar Tidak Dikenali</div>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f'<div class="result-name">{info["nama"]}</div>',
                    unsafe_allow_html=True
                )

                metric1, metric2 = st.columns(2)

                with metric1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence Prediksi</div>
                        <div class="metric-value">{confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metric2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Akurasi Model</div>
                        <div class="metric-value">{MODEL_TEST_ACCURACY:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(float(confidence / 100))

                st.markdown(
                    f'<div class="{status_class}">Tingkat keyakinan model: {status_label}. {status_text}</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

            if info is not None:

                st.markdown('<div class="classification-panel">', unsafe_allow_html=True)
                st.markdown('<div class="result-title">Informasi Kue</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-text">
                    <b>Nama Kue:</b> {info['nama']}<br><br>
                    <b>Asal/Keterangan:</b> {info['asal']}<br><br>
                    <b>Ciri Visual:</b> {info['ciri']}<br><br>
                    <b>Deskripsi:</b> {info['deskripsi']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="classification-panel">
                <div class="result-title">Hasil Prediksi Akan Muncul di Sini</div>
                <div class="info-text">
                    Setelah gambar kue diunggah, sistem akan otomatis menampilkan nama kue,
                    tingkat keyakinan prediksi, informasi kue, top 3 prediksi, dan grafik probabilitas.
                </div>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None and info is not None:
        st.markdown('<div class="classification-panel">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Top 3 Prediksi</div>', unsafe_allow_html=True)

        top_indices = np.argsort(prediction[0])[-3:][::-1]

        for rank, idx in enumerate(top_indices, start=1):
            label = class_names[idx]
            score = float(prediction[0][idx] * 100)
            nama_kue = get_info_kue(label)["nama"]

            st.markdown(f"""
            <div class="prediction-item">
                <div class="prediction-name">{rank}. {nama_kue}</div>
                <div class="prediction-score">{score:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HALAMAN KONTAK
# =========================
elif menu == "Kontak":
    render_hero(
        "Kontak dan Informasi Sistem",
        "Halaman ini berisi informasi singkat mengenai pengembang, teknologi yang digunakan, dan tujuan pembuatan aplikasi.",
        "assets/bg_beranda.jpg",
        "Informasi Pengembang"
    )

    render_section(
        "Informasi Aplikasi",
        "Sistem ini dikembangkan sebagai bagian dari Penulisan Ilmiah dengan fokus pada klasifikasi citra kue tradisional Indonesia."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-card">
            <div class="about-title">Identitas Sistem</div>
            <div class="about-text">
                <b>Nama Sistem:</b> Klasifikasi Citra Kue Tradisional Indonesia<br>
                <b>Metode:</b> Deep Learning<br>
                <b>Arsitektur:</b> MobileNetV2<br>
                <b>Platform:</b> Streamlit<br>
                <b>Format Model:</b> model_kue.keras
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-title">Pengembang</div>
            <div class="about-text">
                <b>Pengembang:</b> Muhammad Ananda Naufal Roffi<br>
                <b>Teknologi:</b> Python, TensorFlow/Keras, MobileNetV2, NumPy, Pillow, dan Streamlit<br>
                <b>Dataset:</b> Kue Indonesia dari Kaggle<br>
                <b>Tujuan:</b> Membantu pengguna mengenali jenis kue tradisional berdasarkan gambar.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    <b>Sistem Klasifikasi Citra Kue Tradisional Indonesia</b><br>
    Menggunakan MobileNetV2 Berbasis Streamlit<br>
    Beranda | Tentang | Nama Makanan | Klasifikasi | Kontak
</div>
""", unsafe_allow_html=True)
