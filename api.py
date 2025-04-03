from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
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

@app.route('/')
def index():
    return jsonify({
        "name": "Voice Cloning API",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/api/clone-voice",
                "method": "POST",
                "description": "Clone a voice using an audio sample and text",
                "parameters": [
                    {"name": "audio", "type": "file", "required": True, "description": "Audio file (.wav, .mp3, .m4a)"},
                    {"name": "text", "type": "string", "required": True, "description": "Text to synthesize"}
                ]
            },
            {
                "path": "/api/health",
                "method": "GET",
                "description": "Check the health of the API"
            }
        ],
        "status": "running"
    })

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
        logger.info(f"Validating audio file: {file.filename}")
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
            logger.error(f"Unsupported audio format: {file.filename}")
            raise ValueError(f"Unsupported audio format: {file.filename}. Supported formats: .wav, .mp3, .m4a")

    @staticmethod
    def process_audio(file):
        logger.info(f"Processing audio file: {file.filename}")
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                file.save(tmp.name)
                logger.info(f"Audio file saved to temporary location: {tmp.name}")

                # Check if file exists and has content
                import os
                if os.path.getsize(tmp.name) == 0:
                    logger.error("Temporary audio file is empty")
                    raise ValueError("Uploaded audio file is empty")

                logger.info("Preprocessing audio file...")
                wav = preprocess_wav(tmp.name)
                logger.info(f"Audio preprocessed, length: {len(wav)} samples")

                if len(wav) < 16000:
                    logger.error(f"Audio too short: {len(wav)} samples")
                    raise ValueError("Audio too short (minimum 1 second)")

                return wav
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}\n{traceback.format_exc()}")
            raise

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    try:
        logger.info("Received voice cloning request")

        # Validate request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({"error": "Audio file required"}), 400

        if 'text' not in request.form:
            logger.error("No text in request")
            return jsonify({"error": "Text required"}), 400

        audio_file = request.files['audio']
        text = request.form['text']

        logger.info(f"Request parameters - Text: '{text}', Audio filename: '{audio_file.filename}'")

        # Process audio
        try:
            AudioProcessor.validate_audio(audio_file)
            wav = AudioProcessor.process_audio(audio_file)
            logger.info(f"Audio processing successful, wav shape: {wav.shape if hasattr(wav, 'shape') else len(wav)}")
        except ValueError as e:
            logger.error(f"Audio validation/processing error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected audio processing error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

        # Check if models are loaded
        if not all([encoder, synthesizer, vocoder_model]):
            logger.error("Models not loaded properly")
            return jsonify({"error": "Voice cloning service not initialized properly"}), 500

        # Generate clone
        logger.info("Generating voice embedding...")
        try:
            # Import necessary functions
            from encoder.audio import wav_to_mel_spectrogram
            from encoder.inference import embed_utterance

            # Convert the audio to mel spectrogram
            logger.info("Converting audio to mel spectrogram...")
            mel = wav_to_mel_spectrogram(wav)
            logger.info(f"Mel spectrogram created, shape: {mel.shape}")

            # Generate the embedding using the global function
            logger.info("Generating embedding from mel spectrogram...")
            embed = embed_utterance(mel, using_partials=False)
            logger.info(f"Embedding generated, shape: {embed.shape if hasattr(embed, 'shape') else len(embed)}")

            logger.info("Synthesizing spectrograms...")
            specs = synthesizer.synthesize_spectrograms([text], [embed])
            logger.info(f"Spectrograms synthesized, shape: {specs[0].shape if hasattr(specs[0], 'shape') else len(specs[0])}")

            logger.info("Inferring waveform...")
            generated_wav = vocoder_model.infer_waveform(specs[0])
            logger.info(f"Waveform generated, length: {len(generated_wav)} samples")
        except Exception as e:
            logger.error(f"Voice synthesis error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"Voice synthesis failed: {str(e)}"}), 500

        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Save and return
        timestamp = int(time.time())
        output_path = output_dir / f"output_{timestamp}.wav"

        try:
            logger.info(f"Saving output to {output_path}")
            sf.write(output_path, generated_wav, 16000)
            logger.info("File saved successfully")

            logger.info("Sending response to client")
            return send_file(
                str(output_path),
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'cloned_voice_{timestamp}.wav'
            )
        except Exception as e:
            logger.error(f"Error saving or sending file: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"Error saving or sending file: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unhandled error in voice cloning: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Voice cloning failed due to an unexpected error"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    status = {
        "status": "ok" if all([encoder, synthesizer, vocoder_model]) else "error",
        "models_loaded": all([encoder, synthesizer, vocoder_model]),
        "timestamp": time.time()
    }
    return jsonify(status)

# Load models on startup
def initialize_models():
    if load_models():
        logger.info("Models loaded successfully on startup")
        return True
    else:
        logger.error("Failed to load models on startup")
        return False

# Initialize models when the module is loaded
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
