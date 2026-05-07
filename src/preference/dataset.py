"""Serializable preference-pair records for offline DPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PreferenceMetadata:
    """Side information attached to each preference pair."""

    utility_difference: float
    preference_mode: str
    source_trajectory_ids: tuple[str, str]
    source_variants: tuple[str, str]
    regime_tag: str
    winner_interval: tuple[float, float]
    loser_interval: tuple[float, float]
    criterion: str


@dataclass
class PreferencePairRecord:
    """Winner/loser trajectory record pair used by continuous DPO."""

    context_id: int
    risk_mode: str
    risk_id: int
    winner_state_tokens: np.ndarray
    winner_lob_states: np.ndarray
    winner_market_returns: np.ndarray
    winner_actions: np.ndarray
    loser_state_tokens: np.ndarray
    loser_lob_states: np.ndarray
    loser_market_returns: np.ndarray
    loser_actions: np.ndarray
    metadata: PreferenceMetadata

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


class PreferenceTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Torch dataset for DPO training."""

    def __init__(self, path: str | Path) -> None:
        self.records = load_preference_dataset(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            "risk_ids": torch.tensor(record.risk_id, dtype=torch.long),
            "winner_tokens": torch.from_numpy(record.winner_state_tokens).long(),
            "winner_actions": torch.from_numpy(record.winner_actions).float(),
            "winner_lob": torch.from_numpy(record.winner_lob_states).float(),
            "winner_returns": torch.from_numpy(record.winner_market_returns).float(),
            "loser_tokens": torch.from_numpy(record.loser_state_tokens).long(),
            "loser_actions": torch.from_numpy(record.loser_actions).float(),
            "loser_lob": torch.from_numpy(record.loser_lob_states).float(),
            "loser_returns": torch.from_numpy(record.loser_market_returns).float(),
        }


def save_preference_dataset(records: list[PreferencePairRecord], path: str | Path) -> None:
    """Persist preference records to pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(records, handle)


def load_preference_dataset(path: str | Path) -> list[PreferencePairRecord]:
    """Load preference records from pickle."""
    with open(path, "rb") as handle:
        return pickle.load(handle)
