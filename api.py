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
import numpy as np

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

# Import necessary modules
from encoder.inference import Encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder
from encoder.audio import preprocess_wav, wav_to_mel_spectrogram

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.set_grad_enabled(False)

# Global model instances
encoder = None
synthesizer = None
vocoder_model = None

# Global flag to track if we've tried to load the vocoder
vocoder_load_attempted = False

def load_vocoder(model_path):
    """
    Load vocoder model with proper state dict handling
    """
    logger.info(f"Loading vocoder from {model_path}")

    try:
        # Try to load the vocoder
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Check if the checkpoint has model_state key (common in training checkpoints)
        if "model_state" in checkpoint:
            # This is likely a training checkpoint with both model and optimizer states
            logger.info("Found model_state in checkpoint, extracting...")
            state_dict = checkpoint["model_state"]
        else:
            # This might be a direct state dict
            logger.info("Using checkpoint directly as state dict...")
            state_dict = checkpoint

        # Initialize your WaveRNN model here
        from vocoder.models.fatchord_version import WaveRNN
        from vocoder.hparams import voc_rnn_dims, voc_fc_dims, bits, voc_pad, voc_upsample_factors, num_mels, voc_compute_dims, voc_res_out_dims, voc_res_blocks, hop_length, sample_rate

        # Use the exact parameters from the vocoder.hparams module
        logger.info(f"Using upsample_factors: {voc_upsample_factors}")
        vocoder = WaveRNN(
            rnn_dims=voc_rnn_dims,
            fc_dims=voc_fc_dims,
            bits=bits,
            pad=voc_pad,
            upsample_factors=voc_upsample_factors,
            feat_dims=num_mels,
            compute_dims=voc_compute_dims,
            res_out_dims=voc_res_out_dims,
            res_blocks=voc_res_blocks,
            hop_length=hop_length,
            sample_rate=sample_rate
        )

        # Load the state dict
        vocoder.load_state_dict(state_dict)
        vocoder.eval()

        return vocoder

    except Exception as e:
        logger.error(f"Error loading vocoder: {str(e)}")
        # If you have a backup or simplified vocoder model, you could try loading that instead
        return None

def load_models():
    global encoder, synthesizer, vocoder_model

    try:
        # Verify models first
        missing = ModelManager.verify_models()
        if missing:
            raise Exception(
                f"Model verification failed. Missing: {missing}"
            )

        logger.info("Loading models...")
        try:
            # Load encoder
            logger.info("Loading encoder model...")
            encoder_path = MODELS_DIR / "encoder.pt"
            logger.info(f"Encoder path: {encoder_path} (exists: {encoder_path.exists()})")
            encoder = Encoder(encoder_path)
            logger.info(f"Encoder loaded successfully: {encoder is not None}")

            # Load synthesizer
            logger.info("Loading synthesizer model...")
            synthesizer_path = MODELS_DIR / "synthesizer.pt"
            logger.info(f"Synthesizer path: {synthesizer_path} (exists: {synthesizer_path.exists()})")
            synthesizer = Synthesizer(synthesizer_path)
            logger.info(f"Synthesizer loaded successfully: {synthesizer is not None}")

            # Load vocoder
            logger.info("Loading vocoder model...")
            vocoder_path = MODELS_DIR / "vocoder.pt"
            logger.info(f"Vocoder path: {vocoder_path} (exists: {vocoder_path.exists()})")
            # Use the global vocoder module
            global vocoder_model

            # Use the new load_vocoder function
            vocoder_model = load_vocoder(vocoder_path)
            logger.info(f"Vocoder loaded successfully: {vocoder_model is not None}")

            # Verify required models are loaded (encoder and synthesizer)
            # Vocoder is optional as we can use Griffin-Lim algorithm instead
            if not encoder or not synthesizer:
                logger.error(f"Required models failed to load. Encoder: {encoder is not None}, Synthesizer: {synthesizer is not None}")
                return False

            # Log vocoder status but don't fail if it's not loaded
            if not vocoder_model:
                logger.warning(f"Vocoder model not loaded. Will use Griffin-Lim algorithm instead.")
        except Exception as e:
            logger.error(f"Error loading specific model: {str(e)}\n{traceback.format_exc()}")
            raise

        logger.info("All models loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}\n{traceback.format_exc()}")
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
                if os.path.getsize(tmp.name) == 0:
                    logger.error("Temporary audio file is empty")
                    raise ValueError("Uploaded audio file is empty")

                logger.info("Preprocessing audio file...")
                wav = preprocess_wav(tmp.name)
                logger.info(f"Audio preprocessed, length: {len(wav)} samples")

                if len(wav) < 16000:
                    logger.error(f"Audio too short: {len(wav)} samples")
                    raise ValueError("Audio too short (minimum 1 second)")

                # Convert to mel spectrogram for the encoder
                logger.info("Converting to mel spectrogram...")
                mel = wav_to_mel_spectrogram(wav)
                logger.info(f"Mel spectrogram created, shape: {mel.shape if hasattr(mel, 'shape') else 'unknown'}")

                return mel
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

        # Check if models are loaded
        if not encoder or not synthesizer:
            logger.error("Encoder or synthesizer models not loaded properly")
            return jsonify({"error": "Voice cloning service not initialized properly"}), 500

        # Check vocoder separately as it might be loaded later
        global vocoder_model, vocoder_load_attempted
        if not vocoder_model:
            if not vocoder_load_attempted:
                logger.warning("Vocoder model not loaded yet, will try to load it on demand")
                # Try to load it now
                try:
                    # First try loading with the module's function
                    vocoder_model = vocoder.load_model(MODELS_DIR / "vocoder.pt")
                    logger.info(f"Loaded vocoder model on demand: {vocoder_model is not None}")

                    # If that didn't work, try direct loading
                    if not vocoder_model:
                        logger.warning("Trying direct torch.load for vocoder model")
                        model_state = torch.load(MODELS_DIR / "vocoder.pt", map_location='cpu')
                        if model_state is not None:
                            # If we can load the state dict, try to initialize the model
                            from vocoder.models.fatchord_version import WaveRNN
                            from vocoder.hparams import voc_rnn_dims, voc_fc_dims, bits, voc_pad, voc_upsample_factors, num_mels, voc_compute_dims, voc_res_out_dims, voc_res_blocks, hop_length, sample_rate
                            vocoder_model = WaveRNN(
                                rnn_dims=voc_rnn_dims,
                                fc_dims=voc_fc_dims,
                                bits=bits,
                                pad=voc_pad,
                                upsample_factors=voc_upsample_factors,
                                feat_dims=num_mels,
                                compute_dims=voc_compute_dims,
                                res_out_dims=voc_res_out_dims,
                                res_blocks=voc_res_blocks,
                                hop_length=hop_length,
                                sample_rate=sample_rate
                            )
                            vocoder_model.load_state_dict(model_state)
                            vocoder_model.eval()
                            logger.info("Successfully loaded vocoder model directly")

                    vocoder_load_attempted = True
                except Exception as e:
                    logger.error(f"Failed to load vocoder model on demand: {str(e)}")
                    vocoder_load_attempted = True

            # If we still don't have a vocoder model, we'll use the synthesizer directly
            # No need to check here, we'll handle it in the synthesis step
            if not vocoder_model:
                logger.warning("Will use synthesizer directly without vocoder")

        # Process audio
        try:
            AudioProcessor.validate_audio(audio_file)
            mel = AudioProcessor.process_audio(audio_file)
            logger.info(f"Audio processing successful, mel shape: {mel.shape if hasattr(mel, 'shape') else 'unknown'}")
        except ValueError as e:
            logger.error(f"Audio validation/processing error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected audio processing error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

        # Generate clone
        logger.info("Generating voice embedding...")
        try:
            # Generate the embedding using the encoder
            logger.info("Generating embedding from mel spectrogram...")
            embed = encoder.embed_utterance(mel)
            logger.info(f"Embedding generated, shape: {embed.shape if hasattr(embed, 'shape') else len(embed)}")

            logger.info("Synthesizing spectrograms...")
            specs = synthesizer.synthesize_spectrograms([text], [embed])
            logger.info(f"Spectrograms synthesized, shape: {specs[0].shape if hasattr(specs[0], 'shape') else len(specs[0])}")

            # Generate waveform using Griffin-Lim directly (skip vocoder)
            logger.info("Inferring waveform using Griffin-Lim...")
            # Use synthesizer's built-in Griffin-Lim algorithm
            try:
                # Try to use the griffin_lim method if it exists
                logger.info("Using synthesizer's griffin_lim method")
                generated_wav = synthesizer.griffin_lim(specs[0])
            except (AttributeError, Exception) as e:
                # If the method doesn't exist or fails, use the audio module directly
                logger.info(f"Griffin-Lim method failed: {str(e)}. Using audio module directly.")
                from synthesizer.audio import inv_mel_spectrogram
                from synthesizer.hparams import hparams
                logger.info("Using audio module's inv_mel_spectrogram function")
                generated_wav = inv_mel_spectrogram(specs[0], hparams)

            logger.info(f"Waveform generated with Griffin-Lim, length: {len(generated_wav)} samples")

            # Normalize the audio to prevent clipping
            max_wav = np.max(np.abs(generated_wav))
            if max_wav > 1.0:
                generated_wav = generated_wav / max_wav * 0.9
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
    # Check if vocoder is loaded, if not try to load it
    global vocoder_model
    if not vocoder_model:
        try:
            import vocoder.inference as vocoder
            vocoder_model = vocoder.load_model(MODELS_DIR / "vocoder.pt")
            logger.info(f"Loaded vocoder model on health check: {vocoder_model is not None}")
        except Exception as e:
            logger.error(f"Failed to load vocoder model on health check: {str(e)}")

    # Prepare status response
    encoder_loaded = encoder is not None
    synthesizer_loaded = synthesizer is not None
    vocoder_loaded = vocoder_model is not None

    # We only need encoder and synthesizer to be loaded
    # Vocoder is optional as we can use Griffin-Lim algorithm instead
    all_required_loaded = encoder_loaded and synthesizer_loaded

    status = {
        "status": "ok" if all_required_loaded else "error",
        "models": {
            "encoder": encoder_loaded,
            "synthesizer": synthesizer_loaded,
            "vocoder": vocoder_loaded,
            "griffin_lim_fallback": True
        },
        "timestamp": time.time()
    }
    return jsonify(status)

# Load models on startup
def initialize_models():
    logger.info("Initializing models on startup...")
    success = load_models()
    if success:
        logger.info("Models loaded successfully on startup")
        # Verify that models are actually loaded
        if all([encoder, synthesizer, vocoder_model]):
            logger.info("All model objects are properly initialized")
            return True
        else:
            logger.error(f"Some model objects are still None after loading. Encoder: {encoder is not None}, Synthesizer: {synthesizer is not None}, Vocoder: {vocoder_model is not None}")
            return False
    else:
        logger.error("Failed to load models on startup")
        return False

# Initialize models when the module is loaded
logger.info("Starting model initialization...")
success = initialize_models()
logger.info(f"Model initialization {'succeeded' if success else 'failed'}")

# If models failed to load, try one more time
if not success:
    logger.info("Attempting to load models again...")
    success = initialize_models()
    logger.info(f"Second model initialization attempt {'succeeded' if success else 'failed'}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
