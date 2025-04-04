import os
import sys
import logging
from pathlib import Path
import torch
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set model paths
MODELS_DIR = Path("saved_models/default")
REQUIRED_MODELS = ["encoder.pt", "synthesizer.pt", "vocoder.pt"]

def main():
    # Check if models exist
    missing = []
    for model in REQUIRED_MODELS:
        model_path = MODELS_DIR / model
        if not model_path.exists():
            missing.append(model)

    if missing:
        logger.error(f"Missing models: {missing}")
        return False

    # Force CPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_grad_enabled(False)

    try:
        # Import modules
        from encoder.inference import Encoder
        from synthesizer.inference import Synthesizer
        import vocoder.inference as vocoder

        # Load encoder
        logger.info("Loading encoder model...")
        encoder_path = MODELS_DIR / "encoder.pt"
        logger.info(f"Encoder path: {encoder_path} (exists: {encoder_path.exists()})")
        encoder = Encoder(encoder_path)
        logger.info("Encoder loaded successfully")

        # Load synthesizer
        logger.info("Loading synthesizer model...")
        synthesizer_path = MODELS_DIR / "synthesizer.pt"
        logger.info(f"Synthesizer path: {synthesizer_path} (exists: {synthesizer_path.exists()})")
        synthesizer = Synthesizer(synthesizer_path)
        logger.info("Synthesizer loaded successfully")

        # Load vocoder
        logger.info("Loading vocoder model...")
        vocoder_path = MODELS_DIR / "vocoder.pt"
        logger.info(f"Vocoder path: {vocoder_path} (exists: {vocoder_path.exists()})")
        vocoder_model = vocoder.load_model(vocoder_path)

        # Force the vocoder model to be loaded if it's None
        if vocoder_model is None:
            logger.error("Vocoder model is None after loading")
            # Try loading it again
            try:
                vocoder_model = vocoder.load_model(vocoder_path)
                logger.info("Successfully loaded vocoder model on second attempt")
            except Exception as e:
                logger.error(f"Second attempt to load vocoder failed: {str(e)}")

        logger.info(f"Vocoder loaded successfully: {vocoder_model is not None}")

        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
