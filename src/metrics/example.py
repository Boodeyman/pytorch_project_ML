import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, metric=None, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Игнорируем metric параметр - используем свою логику

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """Простая accuracy метрика"""
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).float()
        accuracy = correct.mean().item() * 100
        return accuracy
