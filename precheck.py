import sys
import importlib
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "soundfile",
    "librosa",
    "unidecode",
    "flask",
    "flask_cors"
]

MODELS_DIR = Path("saved_models/default")
REQUIRED_MODELS = ["encoder.pt", "synthesizer.pt", "vocoder.pt"]

def check_dependencies():
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    return missing

def check_models():
    missing = []
    for model in REQUIRED_MODELS:
        if not (MODELS_DIR / model).exists():
            missing.append(model)
    
    return missing

def main():
    # Check Python packages
    missing_packages = check_dependencies()
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
        
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check models
    missing_models = check_models()
    if missing_models:
        logger.error(f"Missing models: {missing_models}")
        return False
        
    logger.info("All dependency checks passed!")
    return True

if __name__ == "__main__":
    if not main():
        sys.exit(1)