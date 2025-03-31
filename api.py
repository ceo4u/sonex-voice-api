import numpy as np
np.bool = bool

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import sys
import gdown
import tempfile
import librosa
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment  # For audio conversion
from encoder.audio import preprocess_wav

# Add the Real-Time-Voice-Cloning directory to the Python path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Import the required modules from your RTVC repository
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder
from encoder.audio import preprocess_wav

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Model URLs (replace with your actual URLs)
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

# Download models if necessary
download_models()

# Initialize models (force CPU usage)
print("Loading models...")
device = "cpu"
encoder_model = Encoder(Path(os.path.join("saved_models", "default", "encoder.pt")))
synthesizer_model = Synthesizer(os.path.join("saved_models", "default", "synthesizer.pt"))
vocoder.load_model(Path(os.path.join("saved_models", "default", "vocoder.pt")))
print("Models loaded successfully!")

# Root endpoint
@app.route("/", methods=["GET"])
def home():
    return "Voice Cloning API is running!"

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


# Voice cloning endpoint
@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    try:
        print("\n=== New Request Received ===")
        
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Missing audio file or text"}), 400

        audio_file = request.files['audio']
        text = request.form['text']
        print(f"Input received - Text: '{text}', Audio: {audio_file.filename}")

        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
            print(f"Temporary audio saved to: {temp_audio_path}")

        # Preprocess audio
        print("\n1. Preprocessing audio...")
        wav = preprocess_wav(temp_audio_path)
        print(f"Preprocessed waveform: {len(wav)} samples ({len(wav)/16000:.2f} seconds at 16kHz)")

        # Clean up temporary file
        os.unlink(temp_audio_path)
        print("Cleaned up temporary file")

        # Generate embeddings
        print("\n2. Generating embeddings...")
        embeddings = encoder_model.embed_utterance(wav)
        print(f"Embeddings shape: {embeddings.shape}")

        # Generate spectrogram
        print("\n3. Synthesizing spectrograms...")
        specs = synthesizer_model.synthesize_spectrograms([text], [embeddings])
        print(f"Synthesized spectrogram shape: {specs[0].shape}")

        # Generate waveform
        print("\n4. Generating waveform...")
        generated_wav = vocoder.infer_waveform(specs[0])
        print(f"Generated waveform: {len(generated_wav)} samples ({len(generated_wav)/22050:.2f} seconds at 22.05kHz)")

        def process_voice_cloning(generated_wav):
    try:
        # Save output
        output_filename = f"generated_{os.urandom(4).hex()}.wav"
        output_path = os.path.join("generated_audio", output_filename)
        os.makedirs("generated_audio", exist_ok=True)
        sf.write(output_path, generated_wav, 22050)
        print(f"\nOutput saved to: {output_path}")

        return jsonify({
            "message": "Voice cloning successful",
            "filename": output_filename
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()  
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,  # Keep debug for error pages
        use_reloader=False  # Disable automatic restarts
    )