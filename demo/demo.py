import streamlit as st
from PIL import Image
from itertools import cycle
import os
import sys

# Add parent folder to path so demo.py can find predict.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict import predict_step

# --------------------------
# Sample Images for testing
# --------------------------
SAMPLE_IMAGES = {
    "First": "../sample_images/basketball_player.jpg",
    "Second": "../sample_images/dog_sitting_in_tub.jpg"
}

# --------------------------
# Show sample images
# --------------------------
def show_sample_images():
    cols = cycle(st.columns(3))
    for img_path in SAMPLE_IMAGES.values():
        if not os.path.exists(img_path):
            st.warning(f"File not found: {img_path}")
            continue
        next(cols).image(img_path, width=200)

    for i, img_path in enumerate(SAMPLE_IMAGES.values()):
        if os.path.exists(img_path) and next(cols).button("Predict Caption", key=i):
            captions = predict_step([img_path], False)
            st.write(f"{i+1}. {captions[0]}")

# --------------------------
# Upload images from computer
# --------------------------
def image_uploader():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("Submit")
        if submitted and images:
            captions = predict_step(images, False)
            for i, caption in enumerate(captions):
                st.write(f"{i+1}. {caption}")

# --------------------------
# Input images via URL
# --------------------------
def images_url():
    with st.form("url"):
        urls = st.text_input('Enter URLs of Images (comma separated)')
        submitted = st.form_submit_button("Submit")
        if submitted and urls:
            images = urls.split(',')
            captions = predict_step(images, True)
            for i, caption in enumerate(captions):
                st.write(f"{i+1}. {caption}")

# --------------------------
# Main App
# --------------------------
def main():
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    st.title("Image Caption Prediction")
    st.header("Welcome to Image Caption Prediction!")
    st.write("Upload an image, use a sample image, or provide an image URL to get captions.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Sample Images",]()
