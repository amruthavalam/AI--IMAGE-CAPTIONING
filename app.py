import torch
import torchvision.transforms as transforms
from torchvision import models
from transformers import BlipProcessor, BlipForConditionalGeneration
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load ResNet model for feature extraction
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Load Hugging Face BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image):
    """Extract deep features from image using ResNet50"""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet(image)
    return features

def generate_caption(image):
    """Generate captions using BLIP model"""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = caption_model.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

@app.route('/')
def home():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and return generated caption"""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Extract features (not used directly for captioning)
    _ = extract_features(image)

    # Generate caption
    caption = generate_caption(image)

    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(debug=True)

