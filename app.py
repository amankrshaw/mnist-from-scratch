import streamlit as st
import numpy as np
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas


from neural_network import load_model, predict_digit

# Load model
model = load_model("mnist_model.pkl")

st.title("MNIST Handwritten Digit Recognition")

st.write("Draw a digit in the box below and I will try to guess it!")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=5,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas to grayscale
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8))
        img = img.convert("L")  # grayscale
        img_np = np.array(img)

        pred_num, pred_label = predict_digit(img_np, model)
        st.subheader(f"Prediction: {pred_num} — {pred_label}")
    else:
        st.warning("Please draw something first!")
