# Adapted from:
# Vocos: https://github.com/gemelo-ai/vocos/blob/main/vocos/feature_extractors.py
# BigVGAN: https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py (Also used by HiFT)

import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    return torch.log(torch.clip(x, min=clip_val))


class MelSpectrogramFeature(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        padding: str = "center",
        fmin: int = 0,
        fmax: int | None = None,
        bigvgan_style_mel: bool = False,
    ):
        super().__init__()

        self.bigvgan_style_mel = bigvgan_style_mel
        if bigvgan_style_mel:
            # BigVGAN style: same padding, Slaney mel scale, with normalization
            self.n_fft = n_fft
            self.win_size = n_fft
            self.hop_size = hop_length
            # (n_mels, n_fft // 2 + 1)
            mel_basis = librosa_mel_fn(
                sr=sample_rate, n_fft=n_fft, n_mels=n_mels, norm="slaney", htk=False, fmin=fmin, fmax=fmax
            )
            mel_basis = torch.from_numpy(mel_basis).float()
            hann_window = torch.hann_window(n_fft)
            self.register_buffer("mel_basis", mel_basis)
            self.register_buffer("hann_window", hann_window)
        else:
            # Vocos style: center padding, HTK mel scale, without normalization
            if padding not in ["center", "same"]:
                raise ValueError("Padding must be 'center' or 'same'.")

            self.padding = padding
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                center=padding == "center",
                power=1,
                f_min=fmin,
                f_max=fmax,
            )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mel_specgram (Tensor): Mel spectrogram of the input audio. (B, C, L)
        """
        if self.bigvgan_style_mel:
            return self.bigvgan_mel(audio)
        else:
            return self.vocos_mel(audio)

    def vocos_mel(self, audio: torch.Tensor) -> torch.Tensor:
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")

        specgram = self.mel_spec.spectrogram(audio)
        mel_specgram = self.mel_spec.mel_scale(specgram)

        # Convert to log scale
        mel_specgram = safe_log(mel_specgram)
        return mel_specgram

    def bigvgan_mel(self, audio: torch.Tensor) -> torch.Tensor:
        # Pad so that the output length T = L // hop_length
        padding = (self.n_fft - self.hop_size) // 2
        audio = torch.nn.functional.pad(audio, (padding, padding), mode="reflect")
        audio = audio.reshape(-1, audio.shape[-1])

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.reshape(audio.shape[:-1] + spec.shape[-2:])

        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(self.mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec
