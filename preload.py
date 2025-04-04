import api
print("Models loaded:", api.encoder is not None and api.synthesizer is not None and api.vocoder_model is not None)
