import os
import gdown
import tempfile
from flask import Flask, request, send_file, jsonify
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
from vocoder.inference import Vocoder

app = Flask(__name__)

# Model URLs (replace with your actual Google Drive links)
MODEL_URLS = {
    "encoder": "https://drive.google.com/file/d/1eWOMwjcRstn1N1RHxFCsB5lYOL1HONkY",
    "synthesizer": "https://drive.google.com/file/d/1eWOMwjcRstn1N1RHxFCsB5lYOL1HONkY",
    "vocoder": "https://drive.google.com/file/d/1eWOMwjcRstn1N1RHxFCsB5lYOL1HONkY"
}

def download_models():
    """Download models if missing"""
    os.makedirs("saved_models/default", exist_ok=True)
    
    if not os.path.exists("saved_models/default/encoder.pt"):
        gdown.download(MODEL_URLS["encoder"], "saved_models/default/encoder.pt")
    
    if not os.path.exists("saved_models/default/synthesizer.pt"):
        gdown.download(MODEL_URLS["synthesizer"], "saved_models/default/synthesizer.pt")
        
    if not os.path.exists("saved_models/default/vocoder.pt"):
        gdown.download(MODEL_URLS["vocoder"], "saved_models/default/vocoder.pt")

# Initialize models (auto-download if missing)
download_models()
encoder = Encoder("saved_models/default/encoder.pt")
synthesizer = Synthesizer("saved_models/default/synthesizer.pt")
vocoder = Vocoder("saved_models/default/vocoder.pt")

# ... keep your existing /clone endpoint ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)