import numpy as np
np.bool = bool  # Fix numpy deprecation warning

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import gdown
import tempfile
import soundfile as sf
from synthesizer.audio import Audio
import traceback
from pathlib import Path
from functools import lru_cache

# Add Real-Time-Voice-Cloning to the path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Constants
SAMPLE_RATE = 16000
N_MELS = 80
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
            print(f"Downloading {model_name} model...")
            gdown.download(model_url, model_path, quiet=False, fuzzy=True)

download_models()

# Load models (CPU mode for compatibility)
print("Loading models...")
encoder_model = Encoder(Path(os.path.join("saved_models", "default", "encoder.pt")))
synthesizer_model = Synthesizer(
    os.path.join("saved_models", "default", "synthesizer.pt")
)
vocoder.load_model(Path(os.path.join("saved_models", "default", "vocoder.pt")))
print("Models loaded successfully!")

# Initialize Audio Processor (single instance)
audio_processor = Audio(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=2048,
    hop_length=HOP_LENGTH,
    win_length=1024,
    fmin=0,
    fmax=8000
)
print("Audio processor initialized!")

@app.route("/synthesize", methods=["POST"])
def synthesize():
    temp_file = None
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing text in request"}), 400

        text = data['text']
        print(f"Synthesizing text: '{text}'")

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
        traceback.print_exc()
        return jsonify({"error": f"Synthesis failed: {str(e)}"}), 500
    finally:
        if temp_file:
            os.unlink(temp_file.name)

@app.route("/", methods=["GET"])
def home():
    return "Voice Cloning API is running!"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@lru_cache(maxsize=100)
def clone_voice_cached(text, audio_hash):
    return clone_voice_processing(text, audio_hash)

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    try:
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Missing audio file or text"}), 400

        audio_file = request.files['audio']
        text = request.form['text']
        print(f"Processing request - Text: '{text}', Audio: {audio_file.filename}")

        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            wav = preprocess_wav(temp_audio.name)
            mel_spectrogram = audio_processor.melspectrogram(wav)
            print(f"Mel Spectrogram Shape: {mel_spectrogram.shape}")

        return clone_voice_processing(text, wav)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def clone_voice_processing(text, wav):
    try:
        embeddings = encoder_model.embed_utterance(wav)
        print(f"Embeddings shape: {embeddings.shape}")

        specs = synthesizer_model.synthesize_spectrograms([text], [embeddings])
        generated_wav = vocoder.infer_waveform(specs[0])
        
        output_filename = f"generated_{os.urandom(4).hex()}.wav"
        output_path = os.path.join("generated_audio", output_filename)
        os.makedirs("generated_audio", exist_ok=True)
        sf.write(output_path, generated_wav, SAMPLE_RATE)
        
        return jsonify({
            "message": "Voice cloning successful",
            "filename": output_filename,
            "url": f"/generated_audio/{output_filename}"
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )