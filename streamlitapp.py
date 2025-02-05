import streamlit as st
import requests
import base64
import os
from io import BytesIO
from PIL import Image

# Flask API endpoint
API_URL = "http://127.0.0.1:5000/remove-background"

# Directory to save processed images
SAVE_DIR = "processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure the directory exists

st.title("Background Removal App")
st.write("Upload an image to remove the background using AI.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Send request to Flask API
    with st.spinner("Processing image..."):
        response = requests.post(API_URL, json={"image": img_base64})

    if response.status_code == 200:
        result_data = response.json()
        result_image_base64 = result_data.get("result_image")

        if result_image_base64:
            # Decode base64 to image
            result_image_bytes = base64.b64decode(result_image_base64)
            result_image = Image.open(BytesIO(result_image_bytes))

            # Save image to the custom directory
            file_path = os.path.join(SAVE_DIR, "background_removed.png")
            result_image.save(file_path, format="PNG")

            st.image(result_image, caption="Background Removed", use_column_width=True)

            # Provide download button
            btn = st.download_button(
                label="Download Processed Image",
                data=result_image_bytes,
                file_name="background_removed.png",
                mime="image/png"
            )

            st.success(f"Image saved to: {file_path}")
        else:
            st.error("Failed to process image.")
    else:
        st.error("Error in processing image: " + response.text)
