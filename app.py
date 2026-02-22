import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# Page Config

st.set_page_config(
    page_title="LeafScan — Plant Health AI",
    page_icon="🌿",
    layout="centered"
)


# Custom CSS — Organic Forest Theme

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ---- Root Variables ---- */
:root {
    --forest:   #1a2e1a;
    --moss:     #2d5a27;
    --fern:     #4a7c3f;
    --leaf:     #6ab04c;
    --mint:     #a8d8a0;
    --cream:    #f5f0e8;
    --parchment:#ede5d0;
    --soil:     #8b6914;
    --rust:     #c0392b;
    --amber:    #e67e22;
    --gold:     #f1c40f;
}

/* ---- Global Reset ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream) !important;
    color: var(--forest) !important;
}

.stApp {
    background: 
        radial-gradient(ellipse at 0% 0%, rgba(106,176,76,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(45,90,39,0.10) 0%, transparent 50%),
        var(--cream);
    min-height: 100vh;
}

/* ---- Header ---- */
.leafscan-header {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    position: relative;
}

.leafscan-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 1px;
    height: 2rem;
    background: linear-gradient(to bottom, transparent, var(--fern));
}

.leafscan-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--forest);
    letter-spacing: -1px;
    line-height: 1;
    margin: 0;
}

.leafscan-title span {
    color: var(--leaf);
    font-style: italic;
}

.leafscan-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 300;
    color: var(--fern);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.leaf-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 1.5rem auto;
    color: var(--mint);
    font-size: 1.1rem;
}
.leaf-divider::before,
.leaf-divider::after {
    content: '';
    width: 80px;
    height: 1px;
    background: linear-gradient(to right, transparent, var(--mint));
}
.leaf-divider::before { background: linear-gradient(to right, transparent, var(--mint)); }
.leaf-divider::after  { background: linear-gradient(to left, transparent, var(--mint)); }

/* ---- Upload Zone ---- */
.upload-wrapper {
    background: white;
    border: 2px dashed var(--mint);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.3s;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.upload-wrapper::before {
    content: '🌿';
    position: absolute;
    top: -10px; right: -10px;
    font-size: 4rem;
    opacity: 0.07;
    transform: rotate(20deg);
}

/* ---- Streamlit file uploader override ---- */
[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed var(--mint) !important;
    border-radius: 20px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"] label {
    color: var(--fern) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ---- Image display ---- */
[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(26,46,26,0.15);
}

/* ---- Result Cards ---- */
.result-card {
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    font-family: 'DM Sans', sans-serif;
    animation: fadeSlideUp 0.5s ease-out;
    position: relative;
    overflow: hidden;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-card.healthy {
    background: linear-gradient(135deg, #e8f5e2 0%, #d4edcc 100%);
    border: 1.5px solid var(--leaf);
}

.result-card.diseased {
    background: linear-gradient(135deg, #fdf0e0 0%, #fae0c8 100%);
    border: 1.5px solid var(--amber);
}

.result-card .icon {
    font-size: 3.5rem;
    line-height: 1;
    flex-shrink: 0;
}

.result-card .text-block .label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.6;
}

.result-card .text-block .status {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0.1rem 0;
}

.result-card.healthy .status { color: var(--moss); }
.result-card.diseased .status { color: #a84300; }

.result-card .text-block .disease-name {
    font-size: 0.85rem;
    color: var(--fern);
    margin-top: 0.25rem;
    font-style: italic;
}

/* ---- Confidence Bar ---- */
.confidence-section {
    margin: 1.5rem 0;
    padding: 1.2rem 1.5rem;
    background: white;
    border-radius: 14px;
    border: 1px solid var(--parchment);
}

.confidence-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fern);
    margin-bottom: 0.6rem;
}

.confidence-bar-track {
    background: var(--parchment);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s ease;
}
.confidence-bar-fill.healthy {
    background: linear-gradient(to right, var(--fern), var(--leaf));
}
.confidence-bar-fill.diseased {
    background: linear-gradient(to right, var(--amber), #e74c3c);
}

.confidence-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 0.5rem;
}
.confidence-value.healthy { color: var(--moss); }
.confidence-value.diseased { color: #a84300; }

/* ---- Info tip ---- */
.tip-box {
    background: var(--parchment);
    border-left: 3px solid var(--leaf);
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--forest);
    margin: 1rem 0;
    font-style: italic;
}

/* ---- Footer ---- */
.leafscan-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.75rem;
    color: var(--fern);
    opacity: 0.6;
    font-family: 'DM Sans', sans-serif;
}

/* ---- Streamlit alerts override ---- */
.stAlert {
    border-radius: 14px !important;
    border: none !important;
}

/* ---- Streamlit title / text override ---- */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

/* ---- Spinner ---- */
.stSpinner > div {
    border-top-color: var(--leaf) !important;
}

/* ---- Supported formats hint ---- */
.format-hint {
    font-size: 0.75rem;
    color: var(--fern);
    opacity: 0.65;
    text-align: center;
    margin-top: 0.4rem;
}

/* ---- Tag chip ---- */
.chip {
    display: inline-block;
    background: var(--mint);
    color: var(--forest);
    border-radius: 99px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\chak3\OneDrive\Bureau\Plant-Disease-Detection\plantvillage_cnn.pth"


# Model Architecture

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        return self.dense_layers(out)

# Load Model

@st.cache_resource
def load_model():
    model = CNN(K=16).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# Classes

class_names = [
    'Apple___Apple_scab', 'Apple___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

healthy_classes = [cls for cls in class_names if "healthy" in cls.lower()]


# Transform

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Header
st.markdown("""
<div class="leafscan-header">
    <p class="leafscan-subtitle">AI-powered plant pathology</p>
    <h1 class="leafscan-title">Leaf<span>Scan</span></h1>
    <div class="leaf-divider">🌿</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center; color:#4a7c3f; font-size:0.95rem; margin-bottom:1.5rem;">'
    'Upload a leaf photograph and our model will instantly tell you whether your plant is <strong>Healthy</strong> or <strong>Diseased</strong>.'
    '</p>',
    unsafe_allow_html=True
)

# File Uploader

uploaded_file = st.file_uploader(
    "Drop your leaf image here, or click to browse",
    type=["jpg", "png", "jpeg"],
    label_visibility="visible"
)
st.markdown('<p class="format-hint">Supported formats: JPG · PNG · JPEG</p>', unsafe_allow_html=True)


# Inference & Results

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="", use_column_width=True)

    with st.spinner("🔬 Analysing leaf tissue..."):
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_class = class_names[predicted.item()]
            confidence_pct = confidence.item() * 100

    is_healthy = predicted_class in healthy_classes
    card_class = "healthy" if is_healthy else "diseased"
    status_text = "Healthy" if is_healthy else "Diseased"
    status_icon = "✅" if is_healthy else "⚠️"

    # Result card
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="icon">{"🌿" if is_healthy else "🍂"}</div>
        <div class="text-block">
            <div class="label">Diagnosis</div>
            <div class="status">{status_icon} {status_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    st.markdown(f"""
    <div class="confidence-section">
        <div class="confidence-label">Model Confidence</div>
        <div class="confidence-bar-track">
            <div class="confidence-bar-fill {card_class}" style="width:{confidence_pct:.1f}%"></div>
        </div>
        <div class="confidence-value {card_class}">{confidence_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Contextual tip
    if is_healthy:
        tip = "Great news! No signs of disease were detected. Continue regular monitoring and ensure proper watering and nutrition."
    else:
        tip = "Signs of disease detected. Consider consulting an agronomist and inspecting nearby plants for spread."

    st.markdown(f'<div class="tip-box">💡 {tip}</div>', unsafe_allow_html=True)

# Footer

st.markdown("""
<div class="leafscan-footer">
    LeafScan · Powered by CNN trained on PlantVillage · For research use only
</div>
""", unsafe_allow_html=True)