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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Real-Time-Voice-Cloning to path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Import models with error handling
try:
    from encoder.inference import Encoder
    from synthesizer.inference import Synthesizer
    import vocoder.inference as vocoder
    from encoder.audio import preprocess_wav
    
    # Force CPU mode and disable JIT compilation
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["NO_CUDA"] = "1"
    torch.set_grad_enabled(False)
    
except Exception as e:
    logger.error(f"Failed to import models: {str(e)}")
    raise

app = Flask(__name__)
CORS(app)

# Initialize models
def load_models():
    try:
        logger.info("Loading models...")
        encoder = Encoder(Path("saved_models/default/encoder.pt"))
        synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
        
        # Initialize vocoder with pre-compiled check
        vocoder_model = vocoder.load_model(Path("saved_models/default/vocoder.pt"))
        
        logger.info("All models loaded successfully!")
        return encoder, synthesizer, vocoder_model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

encoder, synthesizer, vocoder_model = load_models()

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    try:
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Audio and text required"}), 400

        audio_file = request.files['audio']
        text = request.form['text']
        
        # Validate audio
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({"error": "Only WAV files supported"}), 400

        # Process audio
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            wav = preprocess_wav(tmp.name)
            
            if len(wav) < 16000:  # 1 second minimum
                return jsonify({"error": "Audio too short (min 1 second)"}), 400

            # Voice cloning pipeline
            embed = encoder.embed_utterance(wav)
            specs = synthesizer.synthesize_spectrograms([text], [embed])
            generated_wav = vocoder_model.infer_waveform(specs[0])
            
            # Save output
            output_path = f"output_{int(time.time())}.wav"
            sf.write(output_path, generated_wav, 16000)
            
            return send_file(output_path, mimetype='audio/wav')

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models_loaded": True})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)