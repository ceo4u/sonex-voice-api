import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import tempfile
import soundfile as sf
import traceback
import time
import logging
from pathlib import Path
import torch
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("saved_models/default")
REQUIRED_MODELS = ["encoder.pt", "synthesizer.pt", "vocoder.pt"]

class ModelManager:
    @staticmethod
    def verify_models():
        missing = []

        for model in REQUIRED_MODELS:
            model_path = MODELS_DIR / model
            if not model_path.exists():
                missing.append(model)

        return missing

    @staticmethod
    def get_file_checksum(filepath):
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global model instances and functions
encoder = None
synthesizer = None
vocoder_model = None
preprocess_wav = None

def load_models():
    global encoder, synthesizer, vocoder_model, preprocess_wav

    try:
        # Verify models first
        missing = ModelManager.verify_models()
        if missing:
            raise Exception(
                f"Model verification failed. Missing: {missing}"
            )

        # Import after verification
        from encoder.inference import Encoder
        from synthesizer.inference import Synthesizer
        import vocoder.inference as vocoder
        from encoder.audio import preprocess_wav

        # Force CPU mode
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        torch.set_grad_enabled(False)

        logger.info("Loading models...")
        encoder = Encoder(MODELS_DIR / "encoder.pt")
        synthesizer = Synthesizer(MODELS_DIR / "synthesizer.pt")
        vocoder_model = vocoder.load_model(MODELS_DIR / "vocoder.pt")

        logger.info("Models loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

class AudioProcessor:
    @staticmethod
    def validate_audio(file):
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
            raise ValueError("Unsupported audio format")

    @staticmethod
    def process_audio(file):
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            file.save(tmp.name)
            wav = preprocess_wav(tmp.name)

            if len(wav) < 16000:
                raise ValueError("Audio too short (minimum 1 second)")

            return wav

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    try:
        # Validate request
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Audio file and text required"}), 400

        audio_file = request.files['audio']
        text = request.form['text']

        # Process audio
        try:
            AudioProcessor.validate_audio(audio_file)
            wav = AudioProcessor.process_audio(audio_file)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Generate clone
        embed = encoder.embed_utterance(wav)
        specs = synthesizer.synthesize_spectrograms([text], [embed])
        generated_wav = vocoder_model.infer_waveform(specs[0])

        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Save and return
        output_path = output_dir / f"output_{int(time.time())}.wav"
        sf.write(output_path, generated_wav, 16000)

        return send_file(
            str(output_path),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='cloned_voice.wav'
        )

    except Exception as e:
        logger.error(f"Processing error: {traceback.format_exc()}")
        return jsonify({"error": "Voice cloning failed"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    status = {
        "status": "ok" if all([encoder, synthesizer, vocoder_model]) else "error",
        "models_loaded": all([encoder, synthesizer, vocoder_model]),
        "timestamp": time.time()
    }
    return jsonify(status)

# Load models on startup
if load_models():
    logger.info("Models loaded successfully on startup")
else:
    logger.error("Failed to load models on startup")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
