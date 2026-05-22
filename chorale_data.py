"""JSB Chorale loading, tokenization, and PyTorch dataset utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from music21 import chord, corpus, note
from torch.utils.data import DataLoader, Dataset

VOICE_NAMES = ["Soprano", "Alto", "Tenor", "Bass"]

Token = tuple[Any, float]
PAD_TOKEN: Token = ("<PAD>", 0.0)
UNK_TOKEN: Token = ("<UNK>", 0.0)


def encode_part(part) -> list[tuple[int | None, float]]:
    """Convert one voice into a list of (pitch, duration) tuples."""
    sequence = []
    for element in part.flatten().notesAndRests:
        duration = float(element.quarterLength)
        if isinstance(element, note.Note):
            sequence.append((element.pitch.midi, duration))
        elif isinstance(element, note.Rest):
            sequence.append((None, duration))
        elif isinstance(element, chord.Chord):
            sequence.append((element.pitches[-1].midi, duration))
    return sequence


def encode_chorale(score) -> list[list[tuple[int | None, float]]]:
    """Return four voice sequences for a SATB chorale score."""
    return [encode_part(part) for part in score.parts]


def voice_name(part, index: int) -> str:
    return part.partName or part.id or VOICE_NAMES[index]


def load_chorales() -> tuple[list[list[list[tuple[int | None, float]]]], list[dict]]:
    """Load four-voice Bach chorales from the music21 corpus."""
    encoded_chorales: list[list[list[tuple[int | None, float]]]] = []
    metadata: list[dict] = []

    for rk_number, score in enumerate(corpus.chorales.Iterator(returnType="stream"), start=1):
        if len(score.parts) != 4:
            continue

        voices = encode_chorale(score)
        encoded_chorales.append(voices)
        metadata.append(
            {
                "riemenschneider": rk_number,
                "title": score.metadata.title,
                "voice_names": [voice_name(part, i) for i, part in enumerate(score.parts)],
                "sequence_lengths": [len(v) for v in voices],
            }
        )

    return encoded_chorales, metadata


def flatten_voice_sequences(
    encoded_chorales: Sequence[Sequence[Sequence[tuple[int | None, float]]]],
    indices: Iterable[int],
) -> list[list[tuple[int | None, float]]]:
    """Expand selected chorales into one sequence per voice."""
    sequences: list[list[tuple[int | None, float]]] = []
    for idx in indices:
        for voice in encoded_chorales[idx]:
            sequences.append(list(voice))
    return sequences


class PitchDurationVocab:
    """Vocabulary mapping (pitch, duration) tuples to integer token ids."""

    def __init__(self) -> None:
        self.token_to_id: dict[Token, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_token: dict[int, Token] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self._next_id = 2

    def add_token(self, token: Token) -> None:
        if token not in self.token_to_id:
            self.token_to_id[token] = self._next_id
            self.id_to_token[self._next_id] = token
            self._next_id += 1

    def build_from_sequences(
        self,
        sequences: Sequence[Sequence[tuple[int | None, float]]],
    ) -> None:
        for sequence in sequences:
            for token in sequence:
                self.add_token(token)

    def encode(self, sequence: Sequence[tuple[int | None, float]]) -> list[int]:
        unk_id = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(token, unk_id) for token in sequence]

    def decode(self, token_ids: Sequence[int]) -> list[Token]:
        return [self.id_to_token[token_id] for token_id in token_ids]

    def token_id(self, token: Token) -> int:
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: Token) -> bool:
        return token in self.token_to_id


@dataclass
class ChoraleSplits:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


def split_chorale_indices(
    num_chorales: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> ChoraleSplits:
    """Split chorale indices into train/val/test sets without overlap."""
    if abs(train_ratio + val_ratio + (1.0 - train_ratio - val_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    indices = list(range(num_chorales))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(train_ratio * num_chorales)
    val_end = train_end + int(val_ratio * num_chorales)

    return ChoraleSplits(
        train_indices=indices[:train_end],
        val_indices=indices[train_end:val_end],
        test_indices=indices[val_end:],
    )


class SlidingWindowDataset(Dataset):
    """Dataset of fixed-length sliding windows for next-token prediction."""

    def __init__(
        self,
        encoded_sequences: Sequence[Sequence[int]],
        window_size: int = 32,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.window_size = window_size
        self.examples: list[list[int]] = []

        for sequence in encoded_sequences:
            if len(sequence) <= window_size:
                continue
            for start in range(len(sequence) - window_size):
                self.examples.append(sequence[start : start + window_size + 1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_dataloaders(
    encoded_chorales: Sequence[Sequence[Sequence[tuple[int | None, float]]]],
    splits: ChoraleSplits,
    vocab: PitchDurationVocab,
    window_size: int = 32,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from chorale splits and vocabulary."""
    split_sequences = {
        "train": flatten_voice_sequences(encoded_chorales, splits.train_indices),
        "val": flatten_voice_sequences(encoded_chorales, splits.val_indices),
        "test": flatten_voice_sequences(encoded_chorales, splits.test_indices),
    }

    encoded_by_split = {
        split_name: [vocab.encode(sequence) for sequence in sequences]
        for split_name, sequences in split_sequences.items()
    }

    loaders = {}
    for split_name, encoded_sequences in encoded_by_split.items():
        dataset = SlidingWindowDataset(encoded_sequences, window_size=window_size)
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return loaders["train"], loaders["val"], loaders["test"]
