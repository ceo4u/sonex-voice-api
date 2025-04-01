import numpy as np
np.bool = bool  # Fix numpy deprecation warning

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import gdown
import tempfile
import soundfile as sf
from synthesizer.audio import Audio
import traceback
from pathlib import Path
from pydub import AudioSegment
from functools import lru_cache

# Add Real-Time-Voice-Cloning to the path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Import cloning models
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder
from encoder.audio import preprocess_wav
from synthesizer.audio import Audio 

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
device = "cpu"
encoder_model = Encoder(Path(os.path.join("saved_models", "default", "encoder.pt")))
synthesizer_model = Synthesizer(
    os.path.join("saved_models", "default", "synthesizer.pt"),
    sample_rate=16000, hparams={"n_mels": 40}  # Ensure correct feature dimensions
)
vocoder.load_model(Path(os.path.join("saved_models", "default", "vocoder.pt")))
print("Models loaded successfully!")

# Initialize Audio Processor
audio_processor = Audio(
    sample_rate=16000,  # Must match your model's expected sample rate
    n_mels=80,         # Number of mel bands (40 for some models)
    n_fft=2048,        # FFT window size
    hop_length=256,    # Frame shift
    win_length=1024    # Window length
)
print("Audio processor initialized!")

@app.route("/", methods=["GET"])
def home():
    return "Voice Cloning API is running!"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@lru_cache(maxsize=100)
def clone_voice_cached(text, audio_hash):
    """Cached version of voice cloning to optimize speed."""
    return clone_voice_processing(text, audio_hash)

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    try:
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Missing audio file or text"}), 400

        audio_file = request.files['audio']
        text = request.form['text']
        print(f"Processing request - Text: '{text}', Audio: {audio_file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        print(f"Saved temp audio at: {temp_audio_path}")

        # Process audio using the Audio class
        wav = audio_processor.load_wav(temp_audio_path)
        mel_spectrogram = audio_processor.melspectrogram(wav)
        print(f"Processed audio with shape: {mel_spectrogram.shape}")
        

        # Convert and preprocess audio
        audio_processor = Audio(
        sampling_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=1024,  # Add if needed
        mel_fmin=0,
        mel_fmax=8000
        )

        wav = preprocess_wav(temp_audio_path)
        mel_spectrogram = audio_processor.melspectrogram(wav)
        print(f"Mel Spectrogram Shape: {mel_spectrogram.shape}")

        # Clean up
        os.unlink(temp_audio_path)

        return clone_voice_processing(text, wav)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def clone_voice_processing(text, wav):
    try:
        embeddings = encoder_model.embed_utterance(wav)
        print(f"Embeddings shape: {embeddings.shape}")

        specs = synthesizer_model.synthesize_spectrograms([text], [embeddings], styles=[None])
        print(f"Synthesized spectrogram shape: {specs[0].shape}")

        generated_wav = vocoder.infer_waveform(specs[0])
        print(f"Generated waveform length: {len(generated_wav)}")

        output_filename = f"generated_{os.urandom(4).hex()}.wav"
        output_path = os.path.join("generated_audio", output_filename)
        os.makedirs("generated_audio", exist_ok=True)
        sf.write(output_path, generated_wav, 22050)
        
        print(f"Output saved to: {output_path}")

        return jsonify({
            "message": "Voice cloning successful",
            "filename": output_filename
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        use_reloader=False
    )