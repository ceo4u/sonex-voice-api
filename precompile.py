import os
import sys
from pathlib import Path

# Add Real-Time-Voice-Cloning to path
RTVC_DIR = os.path.join(os.path.dirname(__file__), "Real-Time-Voice-Cloning")
sys.path.append(RTVC_DIR)

# Force CPU compilation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NO_CUDA"] = "1"

# Pre-compile vocoder
import vocoder.inference as vocoder
vocoder.load_model(Path("saved_models/default/vocoder.pt"))

print("Vocoder successfully pre-compiled!")

