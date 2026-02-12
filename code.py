import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load model
@st.cache_resource
def load_my_model():
    return load_model("keras_Model.h5", compile=False)

model = load_my_model()

# Load labels
class_names = open("labels.txt", "r").readlines()

st.title("Bild Klassifizierer 🤖")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Resize
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert to numpy
    image_array = np.asarray(image)

    # Normalize
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare data
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.subheader("Ergebnis:")
    st.write("Klasse:", class_name)
    st.write("Sicherheit:", round(float(confidence_score) * 100, 2), "%")

