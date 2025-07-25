import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Load model once
model = tf.keras.models.load_model("mnist_model.h5")


# Function to predict digit
def predict_digit(img):
    img = ImageOps.grayscale(img).resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    return np.argmax(prediction)

# Streamlit UI
st.set_page_config(page_title="Digit Recognizer", layout="wide")
st.title("🖊️ Handwritten Digit Recognition")
st.subheader("Draw a digit below and click 'Predict Now'")

# Canvas
stroke_width = st.slider("Stroke Width", 1, 25, 15)
canvas = st_canvas(
    fill_color="rgba(255,165,0,0.3)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction
if st.button("Predict Now"):
    if canvas.image_data is not None:
        # Convert RGBA to grayscale image
        image_data = np.array(canvas.image_data).astype(np.uint8)
        img = Image.fromarray(image_data).convert("RGBA")
        result = predict_digit(img)
        st.success(f"Predicted Digit: **{result}**")
    else:
        st.warning("Please draw a digit first.")
