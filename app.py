"""
Aethera Pixelis - Advanced Digital Image Processing Application
Main Streamlit Application dengan Pipeline Formula
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import html
from datetime import datetime

# Import pipeline modules
from app.components.dnd_uploader import dnd_uploader
from pipeline import SuperResolution4K, SelectiveTextureEnhancement, CreativeFilmicEffect
from pipeline import filters, enhancement, restoration
from utils import io_utils, metrics
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Aethera Pixelis - Image Processing",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS & STYLING
# ============================================================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #050a1a;
        --bg-secondary: #0b1635;
        --bg-glass: rgba(12, 20, 48, 0.68);
        --border-soft: rgba(120, 150, 255, 0.2);
        --accent: #6d8dff;
        --accent-strong: #ff7bd4;
        --text-primary: #f0f4ff;
        --text-secondary: #a8b8e8;
        --success: #56f4b5;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: var(--text-primary);
    }

    body {
        background:
            radial-gradient(circle at 20% 20%, rgba(84, 67, 203, 0.24), transparent 55%),
            radial-gradient(circle at 80% 0%, rgba(255, 99, 198, 0.18), transparent 45%),
            radial-gradient(circle at 50% 85%, rgba(0, 200, 255, 0.14), transparent 50%),
            linear-gradient(135deg, #050a1a 0%, #020611 60%, #01030a 100%) !important;
    }

    div.block-container {
        padding: 0 2.3rem 4rem;
        max-width: 1180px;
    }

    .hero-section {
        position: relative;
        padding: 78px 64px 82px;
        border-radius: 32px;
        margin: 0 0 46px;
        overflow: hidden;
        background:
            radial-gradient(circle at 10% 15%, rgba(255, 152, 0, 0.2), transparent 65%),
            radial-gradient(circle at 95% 20%, rgba(126, 77, 255, 0.25), transparent 60%),
            linear-gradient(135deg, #11244c 0%, #07122e 45%, #03081d 100%);
        box-shadow: 0 32px 90px -40px rgba(44, 113, 255, 0.85);
        border: 1px solid rgba(124, 172, 255, 0.18);
    }

    .hero-orb {
        position: absolute;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.22) 0%, rgba(255, 255, 255, 0) 70%);
        filter: blur(0.4px);
        opacity: 0.8;
    }

    .hero-orb.one { top: -120px; right: -110px; }
    .hero-orb.two { bottom: -160px; left: -140px; background: radial-gradient(circle, rgba(96, 240, 255, 0.18) 0%, rgba(96, 240, 255, 0) 65%); }

    .hero-copy {
        position: relative;
        z-index: 2;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 18px;
        border-radius: 999px;
        background: rgba(109, 141, 255, 0.18);
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 18px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.78rem;
    }

    .hero-title {
        font-size: 2.95rem;
        font-weight: 800;
        margin: 0 0 12px;
        letter-spacing: 0.01em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        max-width: 520px;
        margin-bottom: 28px;
        line-height: 1.55;
    }

    .hero-highlights {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 26px;
    }

    .highlight-pill {
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(109, 141, 255, 0.15);
        color: var(--text-primary);
        font-size: 0.82rem;
        letter-spacing: 0.01em;
        border: 1px solid rgba(120, 150, 255, 0.25);
    }

    .hero-metrics {
        display: flex;
        gap: 14px;
        margin-top: 28px;
        flex-wrap: wrap;
    }

    .metric-chip {
        background: rgba(11, 23, 58, 0.78);
        border: 1px solid rgba(109, 141, 255, 0.28);
        border-radius: 18px;
        padding: 12px 18px;
        min-width: 120px;
        text-align: center;
    }

    .metric-chip .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(189, 203, 255, 0.7);
    }

    .metric-chip .value {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 4px;
    }

    .steps-card {
        display: block;
    }

    .steps-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 18px;
    }

    .step-item {
        border-radius: 18px;
        padding: 20px 18px 22px;
        background: rgba(8, 16, 42, 0.78);
        border: 1px solid rgba(109, 141, 255, 0.18);
        box-shadow: inset 0 0 30px -24px rgba(109, 141, 255, 0.7);
        min-height: 160px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .step-icon {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: linear-gradient(130deg, var(--accent) 0%, var(--accent-strong) 100%);
        color: #06112d;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 16px 24px -18px rgba(255, 123, 212, 0.9);
    }

    .step-title {
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }

    .step-copy {
        color: rgba(196, 207, 255, 0.8);
        font-size: 0.83rem;
        line-height: 1.5;
    }

    .glass-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-soft);
        border-radius: 24px;
        padding: 28px 32px;
        margin-bottom: 30px;
        box-shadow: 0 30px 80px -60px rgba(82, 120, 255, 0.9);
        backdrop-filter: blur(16px);
    }

    .card-title {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 22px;
        display: flex;
        gap: 12px;
        align-items: center;
        color: rgba(198, 211, 255, 0.85);
    }

    .card-title span {
        padding: 6px 16px;
        border-radius: 999px;
        background: rgba(109, 141, 255, 0.18);
        font-size: 0.72rem;
        color: var(--accent);
    }

    .accent-text {
        color: var(--text-secondary);
        font-size: 0.92rem;
        margin-bottom: 20px;
        line-height: 1.6;
    }

    .preset-description {
        background: rgba(18, 31, 66, 0.85);
        border: 1px solid rgba(109, 141, 255, 0.22);
        border-radius: 16px;
        padding: 16px 18px;
        font-size: 0.88rem;
        color: rgba(206, 217, 255, 0.9);
        margin: 20px 0;
        line-height: 1.55;
    }

    .pipeline-summary ul {
        list-style: none;
        padding: 0;
        margin: 0 0 18px;
        display: grid;
        gap: 12px;
    }

    .pipeline-summary li {
        position: relative;
        padding-left: 28px;
        color: rgba(210, 218, 255, 0.85);
        line-height: 1.5;
    }

    .pipeline-summary li:before {
        content: "";
        position: absolute;
        left: 0;
        top: 10px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: linear-gradient(120deg, var(--accent) 0%, var(--accent-strong) 100%);
        box-shadow: 0 0 12px rgba(109, 141, 255, 0.7);
    }

    button[kind="primary"] {
        background: linear-gradient(120deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
        border: none !important;
        color: #07102a !important;
        font-weight: 700 !important;
        border-radius: 999px !important;
        box-shadow: 0 18px 36px -16px rgba(255, 123, 212, 0.8);
    }

    button[kind="secondary"] {
        border-radius: 999px !important;
        border: 1px solid rgba(142, 170, 255, 0.35) !important;
        color: var(--text-secondary) !important;
        background: rgba(10, 18, 45, 0.35) !important;
    }

    .glass-card img {
        border-radius: 18px;
        box-shadow: 0 24px 48px -32px rgba(82, 120, 255, 0.6);
    }

    .log-box {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        background: rgba(8, 14, 38, 0.75);
        border: 1px solid rgba(98, 128, 255, 0.22);
        border-radius: 16px;
        color: rgba(198, 209, 255, 0.9);
    }

    .stDownloadButton button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        background: linear-gradient(120deg, var(--success) 0%, #41d6ff 100%) !important;
        color: #051126 !important;
        border: none !important;
        box-shadow: 0 16px 40px -26px rgba(65, 214, 255, 0.8);
    }

    .stExpander {
        border: 1px solid rgba(109, 141, 255, 0.25);
        border-radius: 18px !important;
        background: rgba(10, 18, 45, 0.52);
        color: var(--text-primary);
    }

    .stExpander summary {
        font-weight: 600;
    }

    .metric-chip.empty {
        opacity: 0.55;
    }

    .result-card {
        margin-top: 18px;
    }

    .footer {
        text-align: center;
        color: rgba(175, 189, 235, 0.65);
        font-size: 0.82rem;
        padding: 28px 0 12px;
    }

    @media (max-width: 1024px) {
        div.block-container {
            padding: 0 1.4rem 3.5rem;
        }

        .hero-section {
            padding: 56px 28px 62px;
        }

        .hero-title {
            font-size: 2.45rem;
        }

        .glass-card {
            padding: 24px;
        }
    }

    @media (max-width: 768px) {
        .hero-section {
            padding: 48px 22px 58px;
        }

        .preview-window {
            margin-top: 28px;
        }

        .hero-highlights {
            gap: 8px;
        }

        .hero-metrics {
            gap: 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'pipeline_log' not in st.session_state:
    st.session_state.pipeline_log = []
if 'uploaded_file_token' not in st.session_state:
    st.session_state.uploaded_file_token = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_presets():
    """Load preset parameter sets from presets.json with safe fallback."""
    try:
        with open('presets.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback default presets to keep the app running on first deploy
        fallback = {
            "conservative": {
                "super_resolution_4k": {
                    "description": "Low risk enhancement with gentle denoise and sharpening.",
                    "scale_factor": 2,
                    "denoise_strength": 0.5,
                    "detail_boost": 0.55,
                    "local_contrast": 0.12,
                    "halo_guard": 0.6,
                    "edge_mix": 0.6
                },
                "creative_filmic_effect": {
                    "description": "Soft filmic toning with minimal grain and bloom.",
                    "tone_strength": 0.22,
                    "shadow_warmth": 0.10,
                    "highlight_cool": 0.08,
                    "micro_contrast": 0.08,
                    "grain": 0.012,
                    "bloom": 0.03,
                    "mix": 0.6,
                    "skin_preserve": 0.5
                },
                "selective_texture_enhancement": {
                    "description": "Subtle texture recovery prioritising smooth areas.",
                    "detail_gain": 0.45,
                    "micro_contrast": 0.08,
                    "blend": 0.6,
                    "texture_threshold": 0.3,
                    "texture_softness": 1.2,
                    "skin_protect": 0.7,
                    "edge_boost": 0.22
                }
            },
            "balanced": {
                "super_resolution_4k": {
                    "description": "Balanced clarity with adaptive denoise and local contrast.",
                    "scale_factor": 3,
                    "denoise_strength": 0.7,
                    "detail_boost": 0.85,
                    "local_contrast": 0.2,
                    "halo_guard": 0.5,
                    "edge_mix": 0.78
                },
                "creative_filmic_effect": {
                    "description": "Cinematic teal–orange with micro-contrast and gentle bloom.",
                    "tone_strength": 0.35,
                    "shadow_warmth": 0.18,
                    "highlight_cool": 0.16,
                    "micro_contrast": 0.12,
                    "grain": 0.025,
                    "bloom": 0.065,
                    "mix": 0.75,
                    "skin_preserve": 0.4
                },
                "selective_texture_enhancement": {
                    "description": "Recover tactile textures while protecting skin.",
                    "detail_gain": 0.6,
                    "micro_contrast": 0.12,
                    "blend": 0.7,
                    "texture_threshold": 0.26,
                    "texture_softness": 1.0,
                    "skin_protect": 0.6,
                    "edge_boost": 0.3
                }
            },
            "aggressive": {
                "super_resolution_4k": {
                    "description": "Maximum detail boost with stronger sharpening (watch halos).",
                    "scale_factor": 4,
                    "denoise_strength": 0.9,
                    "detail_boost": 1.05,
                    "local_contrast": 0.32,
                    "halo_guard": 0.45,
                    "edge_mix": 0.9
                },
                "creative_filmic_effect": {
                    "description": "Stronger toning, crisp micro-contrast, and visible grain/bloom.",
                    "tone_strength": 0.5,
                    "shadow_warmth": 0.24,
                    "highlight_cool": 0.22,
                    "micro_contrast": 0.2,
                    "grain": 0.045,
                    "bloom": 0.1,
                    "mix": 0.9,
                    "skin_preserve": 0.35
                },
                "selective_texture_enhancement": {
                    "description": "Pronounced detail enhancement with moderate skin protection.",
                    "detail_gain": 0.8,
                    "micro_contrast": 0.18,
                    "blend": 0.8,
                    "texture_threshold": 0.22,
                    "texture_softness": 0.9,
                    "skin_protect": 0.5,
                    "edge_boost": 0.4
                }
            }
        }
        st.info("presets.json tidak ditemukan — menggunakan preset bawaan. Tambahkan file 'presets.json' untuk mengkustomisasi.")
        return fallback


PIPELINE_META = {
    'Super Resolution 4K Enhance': {
        'key': 'super_resolution_4k',
        'tagline': 'Upscale to native 4K with structure-aware clarity recovery.',
        'highlights': [
            'Adaptive denoising tuned to your image noise profile',
            'Structure-sensitive detail boost that avoids halos',
            'Smart halo guard to keep edges clean and natural'
        ]
    },
    'Creative Filmic Effect': {
        'key': 'creative_filmic_effect',
        'tagline': 'Modern cinematic grading with teal-orange split toning.',
        'highlights': [
            'Filmic tonal curve with bloom and grain simulation',
            'Skin-aware blending to keep complexion authentic',
            'Micro-contrast tuning for punch without harshness'
        ]
    },
    'Selective Texture Enhancement': {
        'key': 'selective_texture_enhancement',
        'tagline': 'Bring back tactile textures while protecting skin tones.',
        'highlights': [
            'Adaptive texture masks driven by variance analysis',
            'Multi-scale detail fusion for controlled sharpening',
            'Skin protection controls to avoid oversharpening faces'
        ]
    }
}


def add_log(message):
    """Add message ke pipeline log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.pipeline_log.append(f"[{timestamp}] {message}")


def clear_log():
    """Clear pipeline log"""
    st.session_state.pipeline_log = []


def display_pipeline_formula(pipeline_name):
    """Display Formula Pipeline untuk fitur yang dipilih"""
    
    pipelines = {
        'Super Resolution 4K Enhance': """
            **INPUT IMAGE (Low Resolution)**
                    ↓
            **STEP 1: PRE-DENOISING**
            Method: Bilateral Filter
            Formula: I_p = (1/W_p) × Σ I_q × G_σs(|p-q|) × G_σr(|I_p - I_q|)
            Purpose: Reduce noise before upscaling (noise will be 4x amplified)
                    ↓
            **STEP 2: UPSCALING**
            Method: ESRGAN/Lightweight Interpolation
            Scale Factor: 4x (720p → 4K)
            Purpose: Reconstruct high-resolution details
                    ↓
            **STEP 3: POST-SHARPENING**
            Method: Unsharp Mask
            Formula: I_sharpened = I_original + α × (I_original - I_blurred)
            Purpose: Enhance local contrast and edge sharpness
                    ↓
            **OUTPUT IMAGE (4K Resolution)**
        """,
        'Creative Filmic Effect': """
            **INPUT IMAGE (Original)**
                    ↓
            **STEP 1: TONAL MAPPING**
            Method: 3D Look-Up Table (LUT)
            Preset: Cinematic / Teal-Orange Grade
            Purpose: Map RGB values to predefined cinematic palette
                    ↓
            **STEP 2: COLOR CORRECTION**
            Method: Histogram Matching / LAB Color Adjustment
            Target: Teal shadows, Orange highlights
            Purpose: Align color distribution with film aesthetic
                    ↓
            **STEP 3: TEXTURE SIMULATION**
            Method: Film Grain Overlay (Gaussian/Perlin)
            Intensity: 0.05 - 0.25
            Purpose: Add texture for authentic film appearance
                    ↓
            **OUTPUT IMAGE (Filmic Graded)**
        """,
        'Selective Texture Enhancement': """
            **INPUT IMAGE (Original)**
                    ↓
            **STEP 1: DETAIL EXTRACTION**
            Method: Guided Filter
            Formula: Minimize E = Σ(G_i - a_i×I_i - b_i)² + ε(a_i² + b_i²)
            Output: Base Layer + Detail Layer
            Purpose: Separate low and high frequencies
                    ↓
            **STEP 2: ADAPTIVE SHARPENING**
            Method: Adaptive Contrast Enhancement on Detail Layer
            Technique: CLAHE / Local Contrast Boost
            Purpose: Enhance only high-frequency details (texture)
                    ↓
            **STEP 3: FUSION & LOCAL ADJUSTMENT**
            Method: Laplacian Pyramid Fusion
            Purpose: Blend enhanced details back seamlessly
                    ↓
            **OUTPUT IMAGE (Enhanced Texture)**
        """
    }
    
    return pipelines.get(pipeline_name, "Pipeline tidak tersedia")


# ============================================================================
# HERO EXPERIENCE & UPLOAD FLOW
# ============================================================================

presets_data = load_presets()
pipeline_options = list(PIPELINE_META.keys())

if 'pipeline_select' not in st.session_state:
    st.session_state.pipeline_select = pipeline_options[0]

for meta in PIPELINE_META.values():
    preset_state_key = f"{meta['key']}_preset"
    custom_state_key = f"{meta['key']}_custom"
    if preset_state_key not in st.session_state:
        st.session_state[preset_state_key] = 'balanced'
    if custom_state_key not in st.session_state:
        st.session_state[custom_state_key] = False

hero_container = st.container()
with hero_container:
    st.markdown("<section class='hero-section'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-orb one'></div><div class='hero-orb two'></div>", unsafe_allow_html=True)

    hero_cols = st.columns([1], gap="large")

    with hero_cols[0]:
        st.markdown(
            """
            <div class="hero-copy">
                <div class="hero-badge">Aethera Pixelis</div>
                <h1 class="hero-title">Precision Upscaling & Cinematic Finishing</h1>
                <p class="hero-subtitle">Transform everyday captures into gallery-grade visuals with 4K reconstruction, filmic colour science, and texture-smart enhancement pipelines.</p>
                <div class="hero-highlights">
                    <span class="highlight-pill">4K Neural Upscaling</span>
                    <span class="highlight-pill">Cinematic Colour Science</span>
                    <span class="highlight-pill">Texture-Preserving Detail</span>
                    <span class="highlight-pill">Skin-Aware Blending</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = dnd_uploader(label="Upload your image", key="hero_uploader")

        if uploaded_file is not None:
            upload_token = f"{uploaded_file.name}|{uploaded_file.size}"
            if st.session_state.get("uploaded_file_token") != upload_token:
                try:
                    file_bytes = uploaded_file.read()
                    uploaded_file.seek(0)
                    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                except (UnidentifiedImageError, OSError) as exc:
                    st.error("The uploaded file is not a valid image. Please choose a JPG or PNG file.")
                else:
                    st.session_state.uploaded_image = image
                    st.session_state.processed_image = None
                    st.session_state.uploaded_file_token = upload_token
                    clear_log()
                    add_log(f"New image uploaded: {uploaded_file.name}")
        elif st.session_state.get("uploaded_file_token") and st.session_state.get("hero_uploader") is None:
            st.session_state.uploaded_image = None
            st.session_state.processed_image = None
            st.session_state.uploaded_file_token = None
            clear_log()

        if st.session_state.uploaded_image is not None:
            image_cv = io_utils.ImageIO.load_image_from_pil(st.session_state.uploaded_image)
            img_info = io_utils.ImageIO.get_image_info(image_cv)
            size_text = f"{img_info['size_mb']:.2f} MB"
            metrics_html = f"""
                <div class='hero-metrics'>
                    <div class='metric-chip'>
                        <div class='label'>Resolution</div>
                        <div class='value'>{img_info['resolution']}</div>
                    </div>
                    <div class='metric-chip'>
                        <div class='label'>Channels</div>
                        <div class='value'>{img_info['channels']}</div>
                    </div>
                    <div class='metric-chip'>
                        <div class='label'>File Size</div>
                        <div class='value'>{size_text}</div>
                    </div>
                </div>
            """
        else:
            metrics_html = """
                <div class='hero-metrics'>
                    <div class='metric-chip empty'>
                        <div class='label'>Resolution</div>
                        <div class='value'>Awaiting upload</div>
                    </div>
                    <div class='metric-chip empty'>
                        <div class='label'>Channels</div>
                        <div class='value'>Awaiting upload</div>
                    </div>
                    <div class='metric-chip empty'>
                        <div class='label'>File Size</div>
                        <div class='value'>Awaiting upload</div>
                    </div>
                </div>
            """

        st.markdown(metrics_html, unsafe_allow_html=True)

    st.markdown("</section>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='glass-card steps-card'>
            <div class='card-title'>How It Works <span>Follow the Flow</span></div>
            <div class='steps-grid'>
                <div class='step-item'>
                    <div class='step-icon'>1</div>
                    <div class='step-title'>Upload your image</div>
                    <div class='step-copy'>Drop a JPG or PNG into the hero dropzone or click Browse files. Resolution, channels, and file size appear instantly.</div>
                </div>
                <div class='step-item'>
                    <div class='step-icon'>2</div>
                    <div class='step-title'>Pick the right pipeline</div>
                    <div class='step-copy'>Choose between Super Resolution, Creative Filmic, or Selective Texture. Review the summary card to understand the effect.</div>
                </div>
                <div class='step-item'>
                    <div class='step-icon'>3</div>
                    <div class='step-title'>Enhance & download</div>
                    <div class='step-copy'>Fine-tune parameters if needed, hit Process Image, compare results, then grab the HD output via Download Result.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
controls_col, summary_col = st.columns([1.45, 1], gap="large")

params = {}
preset_mode = 'balanced'

with controls_col:
    st.markdown("<div class='glass-card controls-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Pipeline Controls <span>Choose & Tune</span></div>", unsafe_allow_html=True)

    pipeline = st.selectbox(
        "Enhancement pipeline",
        options=pipeline_options,
        key='pipeline_select'
    )

    meta = PIPELINE_META[pipeline]
    preset_state_key = f"{meta['key']}_preset"
    custom_state_key = f"{meta['key']}_custom"

    preset_mode = st.selectbox(
        "Preset intensity",
        options=['conservative', 'balanced', 'aggressive'],
        key=preset_state_key
    )

    preset_payload = presets_data[preset_mode][meta['key']]
    preset_description = preset_payload.get('description', '')
    preset_params = {k: v for k, v in preset_payload.items() if k != 'description'}

    st.markdown(
        f"<div class='preset-description'><strong>{preset_mode.title()} preset</strong> · {preset_description}</div>",
        unsafe_allow_html=True
    )

    custom_override = st.checkbox("Fine-tune parameters", key=custom_state_key)

    if pipeline == 'Super Resolution 4K Enhance':
        if custom_override:
            col_a, col_b = st.columns(2)
            scale_factor = col_a.select_slider(
                "Upscaling factor",
                options=[2, 3, 4],
                value=int(preset_params.get('scale_factor', 4))
            )
            denoise_strength = col_a.slider(
                "Denoise strength",
                min_value=0.2,
                max_value=1.2,
                value=float(preset_params.get('denoise_strength', 0.7)),
                step=0.05
            )
            detail_boost = col_b.slider(
                "Detail boost",
                min_value=0.3,
                max_value=1.2,
                value=float(preset_params.get('detail_boost', 0.85)),
                step=0.05
            )
            local_contrast = col_b.slider(
                "Local contrast",
                min_value=0.0,
                max_value=0.4,
                value=float(preset_params.get('local_contrast', 0.2)),
                step=0.02
            )
            halo_guard = st.slider(
                "Halo guard",
                min_value=0.3,
                max_value=0.9,
                value=float(preset_params.get('halo_guard', 0.5)),
                step=0.05
            )
            edge_mix = st.slider(
                "Edge mix",
                min_value=0.4,
                max_value=1.0,
                value=float(preset_params.get('edge_mix', 0.78)),
                step=0.05
            )
            params = {
                'scale_factor': int(scale_factor),
                'denoise_strength': float(denoise_strength),
                'detail_boost': float(detail_boost),
                'local_contrast': float(local_contrast),
                'halo_guard': float(halo_guard),
                'edge_mix': float(edge_mix)
            }
        else:
            params = preset_params.copy()

    elif pipeline == 'Creative Filmic Effect':
        if custom_override:
            col_a, col_b = st.columns(2)
            tone_strength = col_a.slider(
                "Tone strength",
                min_value=0.0,
                max_value=0.6,
                value=float(preset_params.get('tone_strength', 0.35)),
                step=0.02
            )
            shadow_warmth = col_a.slider(
                "Shadow warmth",
                min_value=0.0,
                max_value=0.3,
                value=float(preset_params.get('shadow_warmth', 0.18)),
                step=0.01
            )
            highlight_cool = col_a.slider(
                "Highlight cool",
                min_value=0.0,
                max_value=0.3,
                value=float(preset_params.get('highlight_cool', 0.16)),
                step=0.01
            )
            micro_contrast = col_b.slider(
                "Micro contrast",
                min_value=0.0,
                max_value=0.3,
                value=float(preset_params.get('micro_contrast', 0.12)),
                step=0.01
            )
            grain = col_b.slider(
                "Film grain",
                min_value=0.0,
                max_value=0.08,
                value=float(preset_params.get('grain', 0.025)),
                step=0.005
            )
            bloom = st.slider(
                "Bloom strength",
                min_value=0.0,
                max_value=0.12,
                value=float(preset_params.get('bloom', 0.065)),
                step=0.005
            )
            mix = st.slider(
                "Look mix",
                min_value=0.4,
                max_value=0.95,
                value=float(preset_params.get('mix', 0.75)),
                step=0.05
            )
            skin_preserve = st.slider(
                "Skin preserve",
                min_value=0.0,
                max_value=0.6,
                value=float(preset_params.get('skin_preserve', 0.4)),
                step=0.05
            )
            params = {
                'tone_strength': float(tone_strength),
                'shadow_warmth': float(shadow_warmth),
                'highlight_cool': float(highlight_cool),
                'micro_contrast': float(micro_contrast),
                'grain': float(grain),
                'bloom': float(bloom),
                'mix': float(mix),
                'skin_preserve': float(skin_preserve)
            }
        else:
            params = preset_params.copy()

    else:  # Selective Texture Enhancement
        if custom_override:
            col_a, col_b = st.columns(2)
            detail_gain = col_a.slider(
                "Detail gain",
                min_value=0.2,
                max_value=0.9,
                value=float(preset_params.get('detail_gain', 0.6)),
                step=0.05
            )
            micro_contrast = col_a.slider(
                "Micro contrast",
                min_value=0.0,
                max_value=0.2,
                value=float(preset_params.get('micro_contrast', 0.12)),
                step=0.01
            )
            blend = col_b.slider(
                "Blend amount",
                min_value=0.3,
                max_value=0.9,
                value=float(preset_params.get('blend', 0.7)),
                step=0.05
            )
            texture_threshold = col_b.slider(
                "Texture threshold",
                min_value=0.1,
                max_value=0.5,
                value=float(preset_params.get('texture_threshold', 0.26)),
                step=0.02
            )
            texture_softness = st.slider(
                "Texture softness",
                min_value=0.5,
                max_value=1.5,
                value=float(preset_params.get('texture_softness', 1.0)),
                step=0.05
            )
            skin_protect = st.slider(
                "Skin protect",
                min_value=0.3,
                max_value=0.9,
                value=float(preset_params.get('skin_protect', 0.6)),
                step=0.05
            )
            edge_boost = st.slider(
                "Edge boost",
                min_value=0.0,
                max_value=0.5,
                value=float(preset_params.get('edge_boost', 0.3)),
                step=0.02
            )
            params = {
                'detail_gain': float(detail_gain),
                'micro_contrast': float(micro_contrast),
                'blend': float(blend),
                'texture_threshold': float(texture_threshold),
                'texture_softness': float(texture_softness),
                'skin_protect': float(skin_protect),
                'edge_boost': float(edge_boost)
            }
        else:
            params = preset_params.copy()

    action_col1, action_col2 = st.columns(2)
    process_button = action_col1.button("🚀 Process Image", use_container_width=True, type="primary")
    clear_button = action_col2.button("🔄 Reset", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with summary_col:
    st.markdown("<div class='glass-card pipeline-summary'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{pipeline}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='accent-text'>{meta['tagline']}</p>", unsafe_allow_html=True)
    summary_list = "".join([f"<li>{point}</li>" for point in meta['highlights']])
    st.markdown(f"<ul>{summary_list}</ul>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='preset-description'><strong>{preset_mode.title()} preset</strong> · {preset_description}</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📋 Pipeline Formula", expanded=False):
    st.markdown(display_pipeline_formula(pipeline))

if clear_button:
    st.session_state.uploaded_image = None
    st.session_state.processed_image = None
    st.session_state.uploaded_file_token = None
    st.session_state.hero_uploader = None
    clear_log()
    st.rerun()

# Image processing logic
if process_button and st.session_state.uploaded_image:
    # Convert PIL ke OpenCV
    input_cv = io_utils.ImageIO.load_image_from_pil(st.session_state.uploaded_image)
    
    with st.spinner("🔄 Processing image... This may take a moment"):
        try:
            add_log(f"Starting {pipeline} pipeline")
            add_log(f"Input resolution: {input_cv.shape[1]}x{input_cv.shape[0]}")
            
            # Convert BGR to RGB untuk processing
            input_rgb = cv2.cvtColor(input_cv, cv2.COLOR_BGR2RGB)
            
            if pipeline == 'Super Resolution 4K Enhance':
                add_log(f"Scale factor: {params['scale_factor']}x")
                add_log("Using optimized edge-aware detail recovery pipeline")
                
                # Call optimized pipeline
                output_rgb = SuperResolution4K.process(input_rgb, params)
                # Convert back to BGR untuk consistency
                output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
                
                add_log(f"Output resolution: {output.shape[1]}x{output.shape[0]}")
            
            elif pipeline == 'Creative Filmic Effect':
                add_log("Using optimized cinematic grading pipeline with split-toning")
                
                # Call optimized pipeline
                output_rgb = CreativeFilmicEffect.process(input_rgb, params)
                # Convert back to BGR
                output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            
            else:  # Selective Texture Enhancement
                add_log("Using optimized multi-scale texture enhancement pipeline")
                
                # Call optimized pipeline
                output_rgb = SelectiveTextureEnhancement.process(input_rgb, params)
                # Convert back to BGR
                output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            
            st.session_state.processed_image = output
            add_log("✅ Pipeline completed successfully!")
            
        except Exception as e:
            add_log(f"❌ Error: {str(e)}")
            st.error(f"Processing error: {str(e)}")

elif process_button and st.session_state.uploaded_image is None:
    st.warning("Upload an image before running the pipeline.")

# Display results
if st.session_state.uploaded_image or st.session_state.processed_image:
    st.markdown("<div class='glass-card result-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Processed Output <span>Before & After</span></div>", unsafe_allow_html=True)

    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        st.subheader("Input Image")
        if st.session_state.uploaded_image:
            st.image(
                st.session_state.uploaded_image,
                use_container_width=True,
                caption="Original Image"
            )
    
    with col_output:
        st.subheader("Output Image")
        if st.session_state.processed_image is not None:
            output_pil = Image.fromarray(
                io_utils.ImageIO.convert_bgr_to_rgb(st.session_state.processed_image)
            )
            st.image(
                output_pil,
                use_container_width=True,
                caption="Processed Image"
            )
            
            # Download button
            buf = io_utils.ImageIO.save_image_to_bytes(
                st.session_state.processed_image,
                format='PNG'
            )
            st.download_button(
                label="⬇️ Download Result",
                data=buf,
                file_name=f"aethera_pixelis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📝 Processing Log", expanded=True):
    if st.session_state.pipeline_log:
        log_text = "\n".join(st.session_state.pipeline_log)
        st.markdown(
            f"<div class='log-box'><pre>{html.escape(log_text)}</pre></div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Processing log will appear here...")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div class="footer">
        <strong>Aethera Pixelis</strong> © 2024 · Built with Streamlit, OpenCV, and PyTorch
    </div>
    """, unsafe_allow_html=True)
