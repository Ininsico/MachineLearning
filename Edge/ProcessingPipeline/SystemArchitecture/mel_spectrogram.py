import torch
import math

class Melspectrogram(torch.nn.Module):
    def __init__(self, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        mel_filterbanks = self._create_mel_filterbanks()
        self.register_buffer("mel_fb", mel_filterbanks)

    def _hz_to_mel(self, hz):
        return 2595 * math.log10(1 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _create_mel_filterbanks(self):
        fft_freqs = torch.linspace(0, self.sr // 2, self.n_fft // 2 + 1)
        mel_points = torch.linspace(
            self._hz_to_mel(0),
            self._hz_to_mel(self.sr // 2),
            self.n_mels + 2
        )
        hz_points = self._mel_to_hz(mel_points)
        bin_points = torch.floor((self.n_fft + 1) * hz_points / self.sr).long()
        fb = torch.zeros(self.n_mels, len(fft_freqs))

        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return fb

    def forward(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=waveform.device),
            return_complex=True
        )
        magnitude = stft.abs()
        mel_spec = torch.matmul(self.mel_fb, magnitude)
        mel_spec = torch.log(mel_spec + 1e-6)
        return mel_spec