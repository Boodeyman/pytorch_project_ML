import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, metric=None, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """Вычисляем EER (Equal Error Rate)"""
        # Получаем вероятности для класса bonafide (0)
        probs = torch.softmax(logits, dim=-1)
        bonafide_scores = probs[:, 0].detach().cpu().numpy()  # Вероятность bonafide
        labels_np = labels.cpu().numpy()

        # Разделяем scores по классам
        bonafide_mask = labels_np == 0
        spoof_mask = labels_np == 1

        bonafide_scores_filtered = bonafide_scores[bonafide_mask]
        spoof_scores_filtered = bonafide_scores[spoof_mask]

        if len(bonafide_scores_filtered) == 0 or len(spoof_scores_filtered) == 0:
            return 50.0  # Если нет одного из классов

        # Вычисляем EER
        return self.compute_eer(bonafide_scores_filtered, spoof_scores_filtered)

    def compute_eer(self, bonafide_scores, spoof_scores):
        """Вычисление EER"""
        scores = np.concatenate([bonafide_scores, spoof_scores])
        # ИСПРАВЛЕНИЕ: bonafide=1, spoof=0 для ROC
        labels = np.concatenate(
            [np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))]
        )

        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)

        try:
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            return eer * 100
        except Exception:
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            return eer * 100
