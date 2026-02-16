import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Lung Cancer Detection using CNN")
st.write("Upload a CT scan image to predict whether it indicates Cancer or No Cancer.")

st.subheader("Or Try a Sample Image")

col1, col2 = st.columns(2)

sample_image = None

with col1:
    if st.button("Use Sample Healthy Image"):
        sample_image = Image.open("sample_images/healthy.jpg").convert("RGB")

with col2:
    if st.button("Use Sample Cancer Image"):
        sample_image = Image.open("sample_images/cancer.jpg").convert("RGB")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_cancer_model.keras", compile=False)
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((50, 50))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif sample_image is not None:
    image = sample_image
else:
    image = None

if image is not None:
    st.image(image, caption="Uploaded Image", width=300)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    st.write("### Prediction Result:")

    if prediction > 0.5:
        st.error(f"Cancer Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"No Cancer Detected (Confidence: {1 - prediction:.2f})")