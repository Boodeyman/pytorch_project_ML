import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ExampleDataset(BaseDataset):
    """
    ASVspoof 2019 Logical Access dataset (заменяет ExampleDataset)
    """

    def __init__(
        self,
        data_dir=None,
        protocol_file=None,
        partition="train",
        max_length=64000,
        input_length=None,
        dataset_length=None,
        n_classes=None,
        name=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir (str): путь к данным ASVspoof (flac файлы)
            protocol_file (str): путь к protocol файлу (.txt)
            partition (str): train/dev/eval
            max_length (int): максимальная длина аудио в сэмплах

            # Оставляем параметры из оригинального ExampleDataset для совместимости
            input_length, dataset_length, n_classes, name - игнорируются
        """
        if data_dir is None or protocol_file is None:
            # Fallback к оригинальной логике для совместимости
            print("Using synthetic data (original ExampleDataset logic)")
            index = self._create_synthetic_index(
                input_length or 1024,
                n_classes or 2,
                dataset_length or 100,
                name or partition,
            )
        else:
            # Используем реальные ASVspoof данные
            self.data_dir = Path(data_dir)
            self.partition = partition
            self.max_length = max_length

            # Парсим protocol файл
            index = self._parse_protocol(protocol_file, partition)

        super().__init__(index, *args, **kwargs)

    def _parse_protocol(self, protocol_file, partition):
        """Парсинг ASVspoof protocol файла"""
        index = []

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id, file_id, system_id, null_field, label = parts[:5]

                    # Фильтруем по партиции
                    if partition == "train" and not file_id.startswith("LA_T"):
                        continue
                    elif partition == "dev" and not file_id.startswith("LA_D"):
                        continue
                    elif partition == "eval" and not file_id.startswith("LA_E"):
                        continue

                    file_path = self.data_dir / f"{file_id}.flac"

                    # Преобразуем метку: bonafide=0, spoof=1
                    binary_label = 0 if label == "bonafide" else 1

                    if file_path.exists():
                        index.append(
                            {
                                "path": str(file_path),
                                "label": binary_label,
                                "file_id": file_id,
                                "speaker_id": speaker_id,
                                "system_id": system_id,
                            }
                        )

        print(f"Loaded {len(index)} files for {partition}")
        return index

    def _create_synthetic_index(self, input_length, n_classes, dataset_length, name):
        """Создание синтетических данных (оригинальная логика)"""
        index = []
        data_path = ROOT_PATH / "data" / "example" / name
        data_path.mkdir(exist_ok=True, parents=True)

        number_of_zeros = int(np.log10(dataset_length)) + 1

        print("Creating Example Dataset")
        for i in tqdm(range(dataset_length)):
            example_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            example_data = torch.randn(input_length)
            example_label = torch.randint(n_classes, size=(1,)).item()
            torch.save(example_data, example_path)

            index.append({"path": str(example_path), "label": example_label})

        write_json(index, str(data_path / "index.json"))
        return index

    def load_object(self, path):
        """Загрузка аудио или тензора"""
        path = Path(path)

        if path.suffix == ".pt":
            # Оригинальная логика для синтетических данных
            return torch.load(path)
        elif path.suffix in [".flac", ".wav"]:
            # Загрузка реального аудио
            waveform, sample_rate = torchaudio.load(path)

            # Берем только первый канал если стерео
            if waveform.size(0) > 1:
                waveform = waveform[0:1, :]
            waveform = waveform.squeeze(0)  # [T]

            # Ресэмплинг до 16kHz если необходимо
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Обрезка или паддинг до фиксированной длины
            if hasattr(self, "max_length") and self.max_length:
                if waveform.size(0) > self.max_length:
                    # Случайная обрезка для train, центральная для остальных
                    if hasattr(self, "partition") and self.partition == "train":
                        start_idx = torch.randint(
                            0, waveform.size(0) - self.max_length + 1, (1,)
                        ).item()
                        waveform = waveform[start_idx : start_idx + self.max_length]
                    else:
                        start_idx = (waveform.size(0) - self.max_length) // 2
                        waveform = waveform[start_idx : start_idx + self.max_length]
                elif waveform.size(0) < self.max_length:
                    # Паддинг нулями
                    padding = self.max_length - waveform.size(0)
                    waveform = torch.nn.functional.pad(waveform, (0, padding))

            return waveform
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
