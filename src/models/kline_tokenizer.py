"""
Hierarchical tokenizer for OHLCV K-line features using MiniBatchKMeans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construct a feature matrix from OHLCV dataframe.

    Features:
    - Log returns between open/high/low/close
    - Volume and dollar volume z-scores
    - Price range ratios
    """
    features = pd.DataFrame(index=df.index)
    close = df["price_close"].replace(0, np.nan)
    features["log_return"] = np.log(close).diff().fillna(0.0)
    features["range_high"] = (df["price_high"] - df["price_open"]).clip(lower=0)
    features["range_low"] = (df["price_open"] - df["price_low"]).clip(lower=0)
    features["range_close"] = (df["price_close"] - df["price_open"])
    features["volume"] = df["volume"]
    features["dollar_volume"] = df.get("dollar_volume", df["price_close"] * df["volume"])
    features["trades_count"] = df.get("trades_count", 0.0)
    features["imbalance"] = (df.get("buy_volume", 0.0) - df.get("sell_volume", 0.0))
    features.fillna(0.0, inplace=True)
    return features.to_numpy(dtype=np.float32)


@dataclass
class TokenizerConfig:
    n_clusters_level1: int = 64
    n_clusters_level2: int = 8
    batch_size: int = 4096
    random_state: Optional[int] = 42


class HierarchicalKMeansTokenizer:
    """
    Applies a two-level MiniBatchKMeans tokenizer to OHLCV-derived features.

    Each token id is calculated as `coarse_id * n_clusters_level2 + fine_id`.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.level1 = MiniBatchKMeans(
            n_clusters=self.config.n_clusters_level1,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state,
        )
        self.level2_centers: list[np.ndarray] = []
        self.scaler = StandardScaler()
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        return self.config.n_clusters_level1 * self.config.n_clusters_level2

    def fit(self, data: pd.DataFrame | np.ndarray) -> "HierarchicalKMeansTokenizer":
        features = self._prepare_features(data)
        features = self.scaler.fit_transform(features)
        coarse_ids = self.level1.fit_predict(features)

        self.level2_centers = []
        for cluster_id in range(self.config.n_clusters_level1):
            cluster_mask = coarse_ids == cluster_id
            if not np.any(cluster_mask):
                coarse_center = self.level1.cluster_centers_[cluster_id]
                replicated = np.tile(coarse_center, (self.config.n_clusters_level2, 1))
                self.level2_centers.append(replicated.astype(np.float32))
                continue

            fine = MiniBatchKMeans(
                n_clusters=self.config.n_clusters_level2,
                batch_size=self.config.batch_size,
                random_state=self.config.random_state,
            )
            fine.fit(features[cluster_mask])
            self.level2_centers.append(fine.cluster_centers_.astype(np.float32))

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Tokenizer is not fitted.")
        features = self._prepare_features(data)
        features = self.scaler.transform(features)
        coarse_ids = self.level1.predict(features)
        fine_ids = np.zeros_like(coarse_ids)
        for cluster_id in range(self.config.n_clusters_level1):
            mask = coarse_ids == cluster_id
            if not np.any(mask):
                continue
            centers = self.level2_centers[cluster_id]
            distances = np.linalg.norm(features[mask][:, None, :] - centers[None, :, :], axis=-1)
            fine_ids[mask] = distances.argmin(axis=1)

        tokens = coarse_ids * self.config.n_clusters_level2 + fine_ids
        return tokens.astype(np.int64)

    def fit_transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.__dict__,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_var": getattr(self.scaler, "var_", np.square(self.scaler.scale_)).tolist(),
            "level1_centers": self.level1.cluster_centers_.tolist(),
            "level2_centers": [centers.tolist() for centers in self.level2_centers],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalKMeansTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text())
        tokenizer = cls(TokenizerConfig(**payload["config"]))
        tokenizer.scaler.mean_ = np.array(payload["scaler_mean"], dtype=np.float32)
        tokenizer.scaler.scale_ = np.array(payload["scaler_scale"], dtype=np.float32)
        tokenizer.scaler.var_ = np.array(payload["scaler_var"], dtype=np.float32)
        tokenizer.scaler.n_features_in_ = tokenizer.scaler.mean_.shape[0]
        tokenizer.scaler.n_samples_seen_ = 0

        tokenizer.level1.cluster_centers_ = np.array(payload["level1_centers"], dtype=np.float32)
        tokenizer.level1._n_threads = 1

        tokenizer.level2_centers = [np.array(centers, dtype=np.float32) for centers in payload["level2_centers"]]

        tokenizer._fitted = True
        return tokenizer

    def _prepare_features(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return build_feature_matrix(data)
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a pandas DataFrame or numpy ndarray")
        return data
