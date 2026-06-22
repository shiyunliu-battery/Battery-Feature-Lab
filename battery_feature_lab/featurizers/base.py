"""Base feature extraction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from battery_feature_lab.schemas import FeatureConfig, FeatureTable


class BaseFeaturizer(ABC):
    """Abstract base class for feature extractors."""

    name: str

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    @abstractmethod
    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        """Extract features from a normalized battery time-series frame."""
