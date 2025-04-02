import numpy as np
np.bool = bool  # Fix numpy deprecation warning

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import gdown
import tempfile
import gc
import soundfile as sf
from synthesizer.audio import Audio
import traceback
from pathlib import Path
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Real-Time-Voice-Cloning to the path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Constants
SAMPLE_RATE = 16000
N_MELS = 40
HOP_LENGTH = 256

# Import cloning models
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder
from encoder.audio import preprocess_wav

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_URLS = {
    "encoder": "https://media.githubusercontent.com/media/ceo4u/sonex-voice-api/main/saved_models/default/encoder.pt",
    "synthesizer": "https://media.githubusercontent.com/media/ceo4u/sonex-voice-api/main/saved_models/default/synthesizer.pt",
    "vocoder": "https://media.githubusercontent.com/media/ceo4u/sonex-voice-api/main/saved_models/default/vocoder.pt"
}

def download_models():
    """Download models if missing."""
    model_dir = os.path.join("saved_models", "default")
    os.makedirs(model_dir, exist_ok=True)
    
    for model_name, model_url in MODEL_URLS.items():
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model_name} model...")
            gdown.download(model_url, model_path, quiet=False, fuzzy=True)

# Create a custom audio processor class that uses the updated librosa methods
class CustomAudio(Audio):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build_mel_basis(self):
        # Use keyword arguments for librosa.filters.mel
        return librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

# Memory management function
def cleanup():
    """Force garbage collection to free memory"""
    gc.collect()

# Download models if needed
try:
    download_models()
except Exception as e:
    logger.error(f"Error downloading models: {str(e)}")
    raise

# Load models (CPU mode for compatibility)
logger.info("Loading models...")
try:
    encoder_model = Encoder(Path(os.path.join("saved_models", "default", "encoder.pt")))
    synthesizer_model = Synthesizer(
        os.path.join("saved_models", "default", "synthesizer.pt")
    )
    vocoder.load_model(Path(os.path.join("saved_models", "default", "vocoder.pt")))
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Initialize Audio Processor (single instance)
try:
audio_processor = Audio(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,  # Now using 40 instead of 80
    n_fft=2048,
    hop_length=HOP_LENGTH,
    win_length=1024,
    fmin=0,
    fmax=8000
)
    logger.info("Audio processor initialized!")
except Exception as e:
    logger.error(f"Error initializing audio processor: {str(e)}")
    raise

@app.route("/synthesize", methods=["POST"])
def synthesize():
    temp_file = None
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing text in request"}), 400

        text = data['text']
        logger.info(f"Synthesizing text: '{text}'")

        # Get speaker embedding (replace with actual embedding loading)
        embeddings = encoder_model.embed_utterance(np.zeros(SAMPLE_RATE))  # Placeholder
        
        # Synthesize spectrogram
        specs = synthesizer_model.synthesize_spectrograms([text], [embeddings])
        spec = specs[0]
        
        # Generate waveform
        generated_wav = vocoder.infer_waveform(spec)
        
        # Save temporary audio file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, generated_wav, SAMPLE_RATE)
            
        return send_file(
            temp_file.name,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='synthesized.wav'
        )
            
    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Synthesis failed: {str(e)}"}), 500
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        cleanup()

@app.route("/", methods=["GET"])
def home():
    return "Voice Cloning API is running!"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/test", methods=["GET"])
def test():
    """Simple test endpoint to verify API is working"""
    return jsonify({
        "status": "ok", 
        "message": "API is running!",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Home page"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/api/clone-voice", "method": "POST", "description": "Clone voice with audio file and text"}
        ]
    })

@lru_cache(maxsize=100)
def clone_voice_cached(text, audio_hash):
    return clone_voice_processing(text, audio_hash)

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    temp_file = None
    try:
        # Validate input
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Missing audio file or text"}), 400

        audio_file = request.files['audio']
        text = request.form['text']
        logger.info(f"Processing request - Text: '{text}', Audio: {audio_file.filename}")

        # Validate file size and type
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({"error": "Only WAV files are supported"}), 400
            
        if len(audio_file.read()) > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({"error": "Audio file too large (max 10MB)"}), 400
        audio_file.seek(0)  # Reset pointer after reading

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_file.save(temp_file.name)
        
        try:
            # Preprocess with correct sample rate and mel bands
            wav = preprocess_wav(temp_file.name)
            if len(wav) < 16000:  # Minimum 1 second of audio
                return jsonify({"error": "Audio too short (minimum 1 second required)"}), 400

            logger.info(f"Preprocessed audio length: {len(wav)/16000:.2f} seconds")
            
            # Generate mel spectrogram with correct dimensions (40 bands)
            mel = audio_processor.melspectrogram(wav)
            logger.info(f"Mel spectrogram shape: {mel.shape}")  # Should be (40, time_steps)

            # Voice cloning process
            embeddings = encoder_model.embed_utterance(wav)
            specs = synthesizer_model.synthesize_spectrograms([text], [embeddings])
            generated_wav = vocoder.infer_waveform(specs[0])
            
            # Save output
            output_filename = f"output_{int(time.time())}.wav"
            output_path = os.path.join("generated_audio", output_filename)
            os.makedirs("generated_audio", exist_ok=True)
            sf.write(output_path, generated_wav, 16000)
            
            return send_file(output_path, mimetype='audio/wav')

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            return jsonify({"error": f"Voice cloning failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error"}), 500
        
    finally:
        # Clean up temp files
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def clone_voice_processing(text, wav):
    try:
        # Generate embedding
        embeddings = encoder_model.embed_utterance(wav)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Generate spectrogram
        specs = synthesizer_model.synthesize_spectrograms([text], [embeddings])
        logger.info(f"Generated spectrogram shape: {specs[0].shape}")
        
        # Generate waveform
        generated_wav = vocoder.infer_waveform(specs[0])
        logger.info(f"Generated waveform: {len(generated_wav)} samples")
        
        # Save output
        output_filename = f"generated_{os.urandom(4).hex()}.wav"
        output_dir = os.path.join(os.getcwd(), "generated_audio")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        sf.write(output_path, generated_wav, SAMPLE_RATE)
        logger.info(f"Output saved to: {output_path}")
        
        # Return response
        return jsonify({
            "message": "Voice cloning successful",
            "filename": output_filename,
            "url": f"/generated_audio/{output_filename}"
        })
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    finally:
        cleanup()

@app.route('/generated_audio/<filename>', methods=['GET'])
def serve_generated_audio(filename):
    """Serve generated audio files"""
    try:
        directory = os.path.join(os.getcwd(), "generated_audio")
        return send_file(
            os.path.join(directory, filename),
            mimetype='audio/wav'
        )
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )