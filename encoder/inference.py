from encoder.params_data import *
from encoder.model import SpeakerEncoder
from encoder.audio import preprocess_wav  # We want to expose this function from here
from matplotlib import cm
from encoder import audio
from pathlib import Path
import numpy as np
import torch

class Encoder:
    def __init__(self, model_path):
        # Force CPU usage by explicitly setting the device to CPU
        self.device = torch.device("cpu")
        try:
            # First try loading as TorchScript model, mapping to CPU
            self.model = torch.jit.load(model_path, map_location=self.device)
        except RuntimeError:
            # If that fails, load as regular PyTorch model on CPU
            self.model = SpeakerEncoder(self.device, self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        
    def embed_utterance(self, wav):
        # If wav is a NumPy array, convert it to a tensor with a batch dimension
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float().unsqueeze(0)
        # Ensure the tensor is on CPU
        wav = wav.to(self.device)
        return self.model(wav)

# Global variables for lazy loading
_model = None  # type: SpeakerEncoder
_device = torch.device("cpu")  # Force CPU

def load_model(weights_fpath: Path, device=None):
    """
    Loads the model in memory. If not explicitly called, it will be run on the first call to embed_frames()
    with the default weights file.
    
    :param weights_fpath: The path to saved model weights.
    :param device: This parameter is ignored; CPU is forced.
    """
    global _model, _device
    _device = torch.device("cpu")  # Always use CPU
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, map_location=_device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    # Use weights_fpath.name by ensuring weights_fpath is a Path object
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))

def is_loaded():
    return _model is not None

def embed_frames_batch(frames_batch):
    """
    Computes embeddings for a batch of mel spectrograms.
    
    :param frames_batch: numpy array of float32 of shape (batch_size, n_frames, n_channels)
    :return: numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed

def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
    partial utterances of <partial_utterance_n_frames> each.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices

def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance.
    """
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed

def embed_speaker(wavs, **kwargs):
    raise NotImplemented()

def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)

    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
