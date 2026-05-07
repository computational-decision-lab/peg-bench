"""Dataset wrappers for VQ-VAE, baseline, rollout and preference training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.bars import slice_time_indices


@dataclass(slots=True)
class DatasetSpec:
    """Shared dataset slicing config."""

    seq_len: int = 96
    stride: int = 50
    target_key: str = "return_20"


def segmented_slice_time_indices(length: int, seq_len: int, stride: int, segment_lengths: np.ndarray | None = None) -> np.ndarray:
    """Return sample boundaries while respecting segment/day boundaries."""
    if segment_lengths is None or len(segment_lengths) == 0:
        return slice_time_indices(length, seq_len, stride)
    indices: list[np.ndarray] = []
    offset = 0
    for segment_length in segment_lengths.astype(np.int64):
        if segment_length > seq_len:
            local = np.arange(seq_len, int(segment_length), stride, dtype=np.int64)
            if len(local) > 0:
                indices.append(local + offset)
        offset += int(segment_length)
    if not indices:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(indices, axis=0)


class LOBAutoencoderDataset(Dataset[torch.Tensor]):
    """Dataset for VQ-VAE training on raw LOB states."""

    def __init__(self, dataset_path: str | Path, max_items: int | None = None) -> None:
        payload = np.load(dataset_path)
        lob = payload["lob"].astype(np.float32, copy=False)
        if max_items is not None:
            lob = lob[:max_items]
        # A tiny fraction of 90d bars contain anomalous top-of-book snapshots.
        # Clean them at the training入口 to avoid poisoning VQ/KMeans fallback.
        lob = np.nan_to_num(lob, nan=0.0, posinf=0.0, neginf=0.0)
        self.lob = torch.from_numpy(lob).float()

    def __len__(self) -> int:
        return int(self.lob.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.lob[index]


class LOBTokenSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sequence dataset backed by discrete VQ codes."""

    def __init__(
        self,
        token_file: str | Path,
        dataset_file: str | Path,
        spec: DatasetSpec | None = None,
    ) -> None:
        self.spec = spec or DatasetSpec()
        token_payload = np.load(token_file)
        data_payload = np.load(dataset_file)
        self.codes = token_payload["codes"].astype(np.int64)
        self.targets = data_payload[self.spec.target_key].astype(np.float32)
        segment_lengths = data_payload["segment_lengths"].astype(np.int64) if "segment_lengths" in data_payload else None
        self.indices = segmented_slice_time_indices(
            len(self.codes),
            self.spec.seq_len,
            self.spec.stride,
            segment_lengths=segment_lengths,
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.indices[index])
        start = end - self.spec.seq_len
        tokens = torch.from_numpy(self.codes[start:end]).long()
        target = torch.tensor(self.targets[end], dtype=torch.float32)
        return tokens, target


class LOBLatentSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sequence dataset backed by continuous latent embeddings."""

    def __init__(
        self,
        token_file: str | Path,
        dataset_file: str | Path,
        spec: DatasetSpec | None = None,
    ) -> None:
        self.spec = spec or DatasetSpec()
        token_payload = np.load(token_file)
        data_payload = np.load(dataset_file)
        self.latents = token_payload["latents"].astype(np.float32)
        self.targets = data_payload[self.spec.target_key].astype(np.float32)
        segment_lengths = data_payload["segment_lengths"].astype(np.int64) if "segment_lengths" in data_payload else None
        self.indices = segmented_slice_time_indices(
            len(self.latents),
            self.spec.seq_len,
            self.spec.stride,
            segment_lengths=segment_lengths,
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.indices[index])
        start = end - self.spec.seq_len
        latents = torch.from_numpy(self.latents[start:end]).float()
        target = torch.tensor(self.targets[end], dtype=torch.float32)
        return latents, target


class LOBRawSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sequence dataset backed by raw LOB tensors."""

    def __init__(
        self,
        dataset_file: str | Path,
        spec: DatasetSpec | None = None,
    ) -> None:
        self.spec = spec or DatasetSpec()
        data_payload = np.load(dataset_file)
        self.lob = np.nan_to_num(data_payload["lob"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.targets = np.nan_to_num(data_payload[self.spec.target_key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        segment_lengths = data_payload["segment_lengths"].astype(np.int64) if "segment_lengths" in data_payload else None
        self.indices = segmented_slice_time_indices(
            len(self.lob),
            self.spec.seq_len,
            self.spec.stride,
            segment_lengths=segment_lengths,
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.indices[index])
        start = end - self.spec.seq_len
        lob = torch.from_numpy(self.lob[start:end]).float()
        target = torch.tensor(self.targets[end], dtype=torch.float32)
        return lob, target
