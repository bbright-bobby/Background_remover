from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForImageSegmentation
import torch
from PIL import Image
from io import BytesIO
import base64
from torchvision import transforms
import logging

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app, origins=["http://localhost:4200"])

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
model = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
model.to(device)
model.eval()

def base64_to_image(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        logging.error(f"Error converting base64 to image: {e}")
        raise ValueError("Invalid base64 image format")

def image_to_base64(image):
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {e}")
        raise ValueError("Error converting image to base64")

@app.route('/remove-background', methods=['POST'])
def remove_background():
    try:
        # Get image data from request
        data = request.get_json()
        base64_image = data['image']
        logging.info("Received image for background removal")

        # Convert base64 to image
        image = base64_to_image(base64_image)
        image.save("original_image.png")  # Save for debugging

        # Convert the image to RGB (model expects RGB)
        image_rgb = image.convert("RGB")

        # Image preprocessing and segmentation (as per the BiRefNet code)
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Image preprocessing
        input_image = transform_image(image_rgb).unsqueeze(0).to(device)
        logging.info("Image preprocessed and tensor created.")

        # Run segmentation model
        logging.info("Starting model inference...")
        with torch.no_grad():
            preds = model(input_image)[-1].sigmoid().cpu()
        logging.info("Model inference completed.")

        # Process the prediction
        pred = preds[0].squeeze()
        logging.debug(f"Prediction shape: {pred.shape}")

        # Resize mask and apply it to image (convert the mask to RGBA)
        mask = transforms.ToPILImage()(pred).convert("L").resize(image.size)

        # Convert original image to RGBA
        image = image.convert("RGBA")

        # Create a 4-channel mask (RGB + alpha channel)
        rgba_mask = Image.new("RGBA", image.size, (0, 0, 0, 255))  # Fully opaque mask
        rgba_mask.putalpha(mask)

        # Apply mask to image
        transparent_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)
        logging.info("Background removed and mask applied.")

        # Convert result image to base64
        result_image = image_to_base64(transparent_image)
        logging.info("Result image converted to base64")

        # Return the result image as JSON response
        return jsonify({'result_image': result_image})

    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
