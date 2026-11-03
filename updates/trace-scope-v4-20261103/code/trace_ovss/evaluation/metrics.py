"""Dependency-light semantic-segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulate dataset-level intersection and union statistics."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64)
        self.predicted = torch.zeros(self.num_classes, dtype=torch.float64)
        self.target = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to("cpu", dtype=torch.long).reshape(-1)
        target = target.detach().to("cpu", dtype=torch.long).reshape(-1)
        valid = target != self.ignore_index
        prediction = prediction[valid]
        target = target[valid]
        valid_classes = (
            (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        prediction = prediction[valid_classes]
        target = target[valid_classes]

        matched = prediction[prediction == target]
        self.intersection += torch.bincount(
            matched,
            minlength=self.num_classes,
        ).double()
        self.predicted += torch.bincount(
            prediction,
            minlength=self.num_classes,
        ).double()
        self.target += torch.bincount(
            target,
            minlength=self.num_classes,
        ).double()
        self.union = self.predicted + self.target - self.intersection

    def compute(self) -> dict[str, float]:
        iou = self._safe_ratio(self.intersection, self.union)
        dice = self._safe_ratio(
            2.0 * self.intersection,
            self.predicted + self.target,
        )
        accuracy = self._safe_ratio(self.intersection, self.target)
        aggregate_accuracy = self._safe_ratio(
            self.intersection.sum().reshape(1),
            self.target.sum().reshape(1),
        )[0]
        return {
            "mIoU": round(float(np.nanmean(iou.numpy())) * 100.0, 2),
            "mDice": round(float(np.nanmean(dice.numpy())) * 100.0, 2),
            "mAcc": round(float(np.nanmean(accuracy.numpy())) * 100.0, 2),
            "aAcc": round(float(aggregate_accuracy) * 100.0, 2),
        }

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor):
        result = torch.full_like(numerator, torch.nan, dtype=torch.float64)
        valid = denominator > 0
        result[valid] = numerator[valid] / denominator[valid]
        return result
