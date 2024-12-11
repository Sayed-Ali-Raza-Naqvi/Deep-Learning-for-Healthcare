import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

st.set_page_config(page_title="Nuclei Segmentation App", layout="centered")

# @st.cache_resource
# def load_seg_model():
#     model = load_model("cell_segmentation_model.keras")
#     return model

@st.cache_resource
def load_seg_model():
    model_path = "cell_segmentation_model.keras"  # Update if using an absolute path
    st.write(f"Attempting to load model from: {os.path.abspath(model_path)}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'cell_segmentation_model.keras' is in the correct location.")
        raise
    except ValueError as e:
        st.error(f"Model loading failed: {e}")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        raise

model = load_seg_model()

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_mask(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    predicted_mask = (prediction[0] > 0.5).astype(np.uint8)
    predicted_mask = cv2.resize(predicted_mask, (image.size[0], image.size[1]))
    return predicted_mask

st.title("Nuclei Segmentation with U-Net")
st.write("Upload a microscopy image, and the model will segment nuclei in the image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Segment Nuclei"):
        with st.spinner("Processing..."):
            mask = predict_mask(image)
        st.success("Segmentation complete!")
        
        st.image(mask, caption="Segmented Mask", use_column_width=True, clamp=True)
