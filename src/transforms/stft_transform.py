import torch
import torch.nn as nn
import torchaudio


class STFTTransform(nn.Module):
    """STFT трансформация для входа в LCNN"""

    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
        normalized=True,
        center=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.normalized = normalized
        self.center = center

        # Преобразование в спектрограмму
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,
            normalized=normalized,
            center=center,
            pad_mode="reflect",
            power=None,  # Комплексная спектрограмма
        )

    def forward(self, waveform):
        """
        Args:
            waveform (Tensor): [B, T] или [T]
        Returns:
            spectrogram (Tensor): [B, 1, F, T] логарифм магнитуды
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        # Получаем комплексную спектрограмму
        spec_complex = self.spectrogram(waveform)  # [B, F, T, 2]

        # Вычисляем магнитуду
        magnitude = torch.sqrt(spec_complex[..., 0] ** 2 + spec_complex[..., 1] ** 2)

        # Логарифм магнитуды
        log_magnitude = torch.log(magnitude + 1e-8)

        # Добавляем канал для CNN: [B, 1, F, T]
        log_magnitude = log_magnitude.unsqueeze(1)

        return log_magnitude


class AudioNormalize(nn.Module):
    """Нормализация аудио"""

    def __init__(self, method="instance"):
        super().__init__()
        self.method = method

    def forward(self, waveform):
        if self.method == "instance":
            # Нормализация по экземпляру
            mean = waveform.mean()
            std = waveform.std()
            return (waveform - mean) / (std + 1e-8)
        elif self.method == "global":
            # Глобальная нормализация [-1, 1]
            return waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        else:
            return waveform
