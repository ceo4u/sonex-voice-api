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
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add Real-Time-Voice-Cloning to path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Constants
SAMPLE_RATE = 16000
N_MELS = 40
HOP_LENGTH = 256

# Import models
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder
from encoder.audio import preprocess_wav

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model variables
encoder_model = None
synthesizer_model = None

def load_models():
    """Load models with error handling"""
    global encoder_model, synthesizer_model
    
    try:
        logger.info("Loading models...")
        encoder_model = Encoder(Path("saved_models/default/encoder.pt"))
        synthesizer_model = Synthesizer(Path("saved_models/default/synthesizer.pt"))
        vocoder.load_model(Path("saved_models/default/vocoder.pt"))
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

# Load models when starting up
load_models()

@app.route('/voice-clone', methods=['POST'])
def voice_clone():
    """Main endpoint for voice cloning"""
    temp_file = None
    try:
        # Validate input
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Missing audio or text"}), 400

        audio_file = request.files['audio']
        text = request.form['text']
        logger.info(f"Processing: '{text}', {audio_file.filename}")

        # Validate file
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({"error": "Only WAV files supported"}), 400

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_file.save(temp_file.name)

        try:
            # Preprocess audio
            wav = preprocess_wav(temp_file.name)
            if len(wav) < SAMPLE_RATE:
                return jsonify({"error": "Audio too short (min 1 second)"}), 400

            # Generate embedding
            embed = encoder_model.embed_utterance(wav)
            logger.info("Embedding generated")

            # Synthesize spectrogram
            specs = synthesizer_model.synthesize_spectrograms([text], [embed])
            spec = specs[0]
            logger.info(f"Spectrogram shape: {spec.shape}")

            # Generate waveform
            generated_wav = vocoder.infer_waveform(spec)
            logger.info(f"Waveform length: {len(generated_wav)}")

            # Save output
            output_path = f"output_{int(time.time())}.wav"
            sf.write(output_path, generated_wav, SAMPLE_RATE)
            
            return send_file(output_path, mimetype='audio/wav')

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": "Internal error"}), 500

    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Service healthy"})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "endpoints": {
            "/voice-clone": "POST - Process voice cloning",
            "/health": "GET - Service health check"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)