import os
import sys
import gdown
import tempfile
import librosa
import soundfile as sf
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# Add the Real-Time-Voice-Cloning directory to the Python path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Import the required modules from your RTVC repository
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Model URLs (replace with your actual Google Drive links formatted for direct download)
MODEL_URLS = {
    "encoder": "https://drive.google.com/uc?id=1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1",
    "synthesizer": "https://drive.google.com/uc?id=1EqFMIbvxffxtjiVrtykroF6_mUh-5Z3s",
    "vocoder": "https://drive.google.com/uc?id=1cf2NO6FtI0jDuy8AV3Xgn6leO6dHjIgu"
}

def download_models():
    """Download models if missing."""
    model_dir = os.path.join("saved_models", "default")
    os.makedirs(model_dir, exist_ok=True)
    
    encoder_path = os.path.join(model_dir, "encoder.pt")
    synthesizer_path = os.path.join(model_dir, "synthesizer.pt")
    vocoder_path = os.path.join(model_dir, "vocoder.pt")
    
    if not os.path.exists(encoder_path):
        print("Downloading encoder model...")
        gdown.download(MODEL_URLS["encoder"], encoder_path, quiet=False, fuzzy=True)
    
    if not os.path.exists(synthesizer_path):
        print("Downloading synthesizer model...")
        gdown.download(MODEL_URLS["synthesizer"], synthesizer_path, quiet=False, fuzzy=True)
        
    if not os.path.exists(vocoder_path):
        print("Downloading vocoder model...")
        gdown.download(MODEL_URLS["vocoder"], vocoder_path, quiet=False, fuzzy=True)

# Download models if necessary
download_models()

# Initialize models
print("Loading models...")
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or (sys.platform != "win32" and os.system("nvidia-smi") == 0) else "cpu"
encoder_model = Encoder(os.path.join("saved_models", "default", "encoder.pt"))
synthesizer_model = Synthesizer(os.path.join("saved_models", "default", "synthesizer.pt"))
vocoder.load_model(os.path.join("saved_models", "default", "vocoder.pt"))
print("Models loaded successfully!")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ready"})

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

        # Process the audio file
        wav = librosa.load(temp_audio_path, sr=22050)[0]
        os.unlink(temp_audio_path)  # Clean up temporary file

        # Generate embeddings
        embeddings = encoder_model.embed_utterance(wav)

        # Generate spectrogram
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

@app.route('/api/download/<filename>', methods=['GET'])
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
    app.run(host='0.0.0.0', port=port)
