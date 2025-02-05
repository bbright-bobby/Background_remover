import streamlit as st
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline
import torch
import io
import random

# Load the Stable Diffusion model globally for efficiency
@st.cache_resource
def load_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Function to generate a bubble-shaped graphical background
def generate_bubble_background(width, height):
    background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(background)

    for _ in range(random.randint(30, 50)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(20, 100)
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(120, 200),
        )
        
        draw.ellipse([left_up, right_down], fill=color)

    return background

# Function to generate an AI background using Stable Diffusion
def generate_ai_background(prompt, pipe):
    image = pipe(prompt).images[0]
    return image

# Function to combine the product with the new background
def compose_images(foreground, background):
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")
    background = background.resize(foreground.size, Image.LANCZOS)
    final_image = Image.alpha_composite(background, foreground)
    return final_image

# Streamlit app interface
st.title("AI Background Generator for Product Images")
st.write("Upload a product image and generate a new background!")

# Add file uploader widget
uploaded_file = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image preview
    image = Image.open(uploaded_file).convert("RGBA")
    st.image(image, caption="Uploaded Image Preview", use_container_width=True)
    
    # Select the background style
    style = st.radio("Select a Background Style:", ("Graphical Background", "AI-Generated Background"))
    
    # Generate the background
    if style == "Graphical Background":
        st.write("Generating bubble-shaped background...")
        bg_image = generate_bubble_background(image.width, image.height)
        st.image(bg_image, caption="Bubble-Shaped Background", use_container_width=True)
    else:
        prompt = st.text_input("Enter a prompt for the AI background (e.g., 'modern grocery store')", "modern grocery store")
        if st.button("Generate AI Background"):
            st.write("Generating AI background...")
            pipe = load_stable_diffusion()
            bg_image = generate_ai_background(prompt, pipe)
            st.image(bg_image, caption="AI-Generated Background", use_container_width=True)
    
    # Compose the final image
    if 'bg_image' in locals():
        st.write("Composing the final image...")
        final_image = compose_images(image, bg_image)
        st.image(final_image, caption="Final Composed Image", use_container_width=True)
        
        # Download button
        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        st.download_button(
            label="Download Final Image",
            data=buf.getvalue(),
            file_name="final_composed_image.png",
            mime="image/png"
        )
