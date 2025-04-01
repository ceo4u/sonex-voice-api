import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

# Fix numpy complex type
np.complex = complex

class Audio:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=2048, hop_length=None, win_length=None, 
                 fmin=0, fmax=8000, ref_level_db=20, preemphasis=0.97):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.ref_level_db = ref_level_db
        self.preemphasis = preemphasis
        
    def load_wav(self, path):
        return librosa.core.load(path, sr=self.sample_rate)[0]
    
    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.sample_rate, wav.astype(np.int16))
    
    def melspectrogram(self, wav):
        """Convert waveform to mel spectrogram"""
        D = self._stft(self._preemphasis(wav))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)
    
    def inv_mel_spectrogram(self, mel_spectrogram):
        """Convert mel spectrogram to waveform"""
        D = self._denormalize(mel_spectrogram)
        S = self._mel_to_linear(self._db_to_amp(D + self.ref_level_db))
        return self._inv_preemphasis(self._griffin_lim(S))
    
    def _preemphasis(self, wav):
        return signal.lfilter([1, -self.preemphasis], [1], wav)
    
    def _inv_preemphasis(self, wav):
        return signal.lfilter([1], [1, -self.preemphasis], wav)
    
    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, 
                          win_length=self.win_length)
    
    def _istft(self, y):
        return librosa.istft(y, hop_length=self.hop_length, win_length=self.win_length)
    
    def _build_mel_basis(self):
        return librosa.filters.mel(self.sample_rate, self.n_fft, 
                                 n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
    
    def _linear_to_mel(self, spectrogram):
        mel_basis = self._build_mel_basis()
        return np.dot(mel_basis, spectrogram)
    
    def _mel_to_linear(self, mel_spectrogram):
        mel_basis = self._build_mel_basis()
        inv_mel_basis = np.linalg.pinv(mel_basis)
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spectrogram))
    
    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))
    
    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)
    
    def _normalize(self, S):
        return np.clip((S + 100) / 100, 0, 1)
    
    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * 100) - 100
    
    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex128)
        y = self._istft(S_complex * angles)
        for _ in range(60):  # Griffin-Lim iterations
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

# Helper functions (keep these at the bottom if needed)
def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)