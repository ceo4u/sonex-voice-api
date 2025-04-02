import numpy as np
np.bool = bool  # Fix numpy deprecation warning

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
from encoder.audio import preprocess_wav

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Real-Time-Voice-Cloning to path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Import models
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder

app = Flask(__name__)
CORS(app)

# Load models
logger.info("Loading models...")
encoder = Encoder(Path("saved_models/default/encoder.pt"))
synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
vocoder.load_model(Path("saved_models/default/vocoder.pt"))
logger.info("Models loaded!")

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    try:
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Need both audio and text"}), 400

        audio_file = request.files['audio']
        text = request.form['text']
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            wav = preprocess_wav(tmp.name)
            
            # Process voice cloning
            embed = encoder.embed_utterance(wav)
            specs = synthesizer.synthesize_spectrograms([text], [embed])
            generated_wav = vocoder.infer_waveform(specs[0])
            
            # Save output
            output_path = f"output_{int(time.time())}.wav"
            sf.write(output_path, generated_wav, 16000)
            
            return send_file(output_path, mimetype='audio/wav')
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)