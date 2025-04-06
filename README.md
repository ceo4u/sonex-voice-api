# Sonex Voice Cloning API

Sonex Voice Cloning API is a deep learning-powered service that allows you to generate synthetic speech from any person's voice using just a short audio sample. Built with Flask and PyTorch, this backend service can be connected with a frontend interface for real-time voice cloning and text-to-speech synthesis.

---

## 🔥 Features

- 🎙️ Clone any voice using a short WAV sample.
- 🗣️ Generate speech in the cloned voice from text.
- 🧠 Powered by deep learning models for natural synthesis.
- 🌐 REST API support for easy frontend integration.
- ☁️ Deployable on Render, Docker, or any cloud service.

---

## 🛠️ Tools, Libraries, and Frameworks

### Tools
- Git & GitHub
- Docker
- Render (for live deployment)

### Libraries
- Python 3
- NumPy
- librosa
- torch
- scipy
- soundfile
- tqdm
- flask
- scikit-learn
- matplotlib

### Frameworks
- Flask (for API)
- PyTorch (for voice synthesis models)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ceo4u/sonex-voice-api.git
cd sonex-voice-api

2. Create a virtual environment (Optional but recommended)

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Download pretrained models-
bash download_models.sh

5. Start the server-
python api.py

Or with Docker:-
docker build -t sonex-api .
docker run -p 5000:5000 sonex-api

📡 API Endpoints
GET /health
Check if the API is up and running.

POST /clone-voice
Request Form-Data:

audio: WAV file of speaker (reference voice)

text: The text to synthesize into cloned voice

Response:

A .wav file of the synthesized voice.

🧪 Test API with Python
python
import requests

url = "https://sonex-voice-api-.onrender.com/clone-voice"
files = {'audio': open("sample.wav", "rb")}
data = {'text': "Welcome to Sonex Voice Cloning."}

response = requests.post(url, files=files, data=data)

with open("output.wav", "wb") as f:
    f.write(response.content)

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

