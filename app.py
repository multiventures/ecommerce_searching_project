from flask import Flask, request, jsonify
from PIL import Image
import os

from models.blip_model import ImageAnalyzer
from models.nlp_model import TextAnalyzer

app = Flask(__name__)
image_analyzer = ImageAnalyzer()
text_analyzer = TextAnalyzer()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Flask app is running with image and NLP pipeline!"

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
        description = image_analyzer.describe_image(image)
        nlp_data = text_analyzer.analyze_text(description)

        return jsonify({
            "description": description,
            "nlp_analysis": nlp_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
