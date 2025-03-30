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
from pathlib import Path  # Ensure Path is imported

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
# Use the Encoder constructor and pass a Path object for the encoder model file.
encoder_model = Encoder(Path(os.path.join("saved_models", "default", "encoder.pt")))
synthesizer_model = Synthesizer(os.path.join("saved_models", "default", "synthesizer.pt"))
# If you need to load the vocoder with its load_model function, pass a Path object:
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
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({"error": "Missing audio file or text"}), 400

        audio_file = request.files['audio']
        text = request.form['text']

        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Process the audio file using the preprocess_wav function
        from pathlib import Path  # Ensure Path is imported
        wav = preprocess_wav(Path(temp_audio_path))
        os.unlink(temp_audio_path)  # Clean up temporary file

        # Load waveform and convert to mel spectrogram with 40 mel bins
        waveform, sr = librosa.load(temp_audio_path, sr=22050)  # You might not need this line anymore if preprocess_wav returns what you need
        # Instead, if preprocess_wav returns the processed mel spectrogram, use that output directly:
        # For example: processed_input = preprocess_wav(Path(temp_audio_path))

        # Generate embeddings using the processed mel spectrogram
        embeddings = encoder_model.embed_utterance(wav)

        # Generate spectrogram from text and embeddings
        specs = synthesizer_model.synthesize_spectrograms([text], [embeddings])

        # Generate waveform using the vocoder model
        generated_wav = vocoder.infer_waveform(specs[0])

        # Save the generated audio file
        output_filename = f"generated_{os.urandom(4).hex()}.wav"
        output_path = os.path.join("generated_audio", output_filename)
        os.makedirs("generated_audio", exist_ok=True)
        sf.write(output_path, generated_wav, 22050)

        return jsonify({
            "message": "Voice cloning successful",
            "filename": output_filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Download endpoint
@app.route("/api/download/<filename>", methods=["GET"])
def download_file(filename):
    try:
        return send_file(
            os.path.join("generated_audio", filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
