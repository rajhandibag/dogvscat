import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# -------------------------
# Load the trained model
# -------------------------
MODEL_PATH = "cat_dog_model.keras"

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------
# Preprocessing Function
# -------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------
# Prediction Function
# -------------------------
def predict(image):
    prediction = model.predict(image, verbose=0)[0]
    prediction_value = prediction if np.isscalar(prediction) else prediction[0]
    label = 1 if prediction_value > 0.5 else 0
    confidence = prediction_value if label == 1 else 1 - prediction_value
    return {"label": label, "confidence": float(confidence)}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #4B8BBE;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: gray;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🐶🐱 Cat vs Dog Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to find out whether it’s a cat or a dog!</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")
    st.write("### 🔍 Processing...")

    processed_image = preprocess_image(image)
    with st.spinner("Classifying..."):
        result = predict(processed_image)

    label = "Dog 🐶" if result["label"] == 1 else "Cat 🐱"
    confidence = result["confidence"] * 100

    # Nice colored output box
    if result["label"] == 1:
        st.markdown(f"<h2 style='color: green;'>✅ Prediction: {label}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color: blue;'>✅ Prediction: {label}</h2>", unsafe_allow_html=True)

    # Confidence progress bar
    st.write("### Confidence Level")
    st.progress(confidence / 100)
    st.write(f"{confidence:.2f}%")

else:
    st.info("⬆️ Upload a JPG or PNG image to start classification.")
