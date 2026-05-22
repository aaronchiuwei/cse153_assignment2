"""Soprano-conditioned LSTM harmonization (predict alto, tenor, bass)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from music21 import stream
from torch.utils.data import DataLoader, Dataset

from chorale_data import ChoraleSplits, PitchDurationVocab, VOICE_NAMES
from chorale_model import export_midi, tokens_to_part

# Voice indices in encoded chorales: 0=Soprano, 1=Alto, 2=Tenor, 3=Bass
SOPRANO_IDX = 0
LOWER_VOICE_INDICES = (1, 2, 3)


def align_chorale_voices(
    encoded_voices: Sequence[Sequence[int]],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Truncate SATB token sequences to a common length (shortest voice)."""
    length = min(len(v) for v in encoded_voices)
    return (
        encoded_voices[0][:length],
        encoded_voices[1][:length],
        encoded_voices[2][:length],
        encoded_voices[3][:length],
    )


class HarmonizationDataset(Dataset):
    """Sliding windows: soprano context -> alto/tenor/bass tokens at same positions."""

    def __init__(
        self,
        chorale_token_lists: Sequence[Sequence[Sequence[int]]],
        window_size: int = 32,
    ) -> None:
        self.window_size = window_size
        self.examples: list[tuple[list[int], list[int], list[int], list[int]]] = []

        for voices in chorale_token_lists:
            sop, alto, tenor, bass = align_chorale_voices(voices)
            if len(sop) <= window_size:
                continue
            for start in range(len(sop) - window_size):
                end = start + window_size
                self.examples.append(
                    (
                        sop[start:end],
                        alto[start:end],
                        tenor[start:end],
                        bass[start:end],
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sop, alto, tenor, bass = self.examples[idx]
        return (
            torch.tensor(sop, dtype=torch.long),
            torch.tensor(alto, dtype=torch.long),
            torch.tensor(tenor, dtype=torch.long),
            torch.tensor(bass, dtype=torch.long),
        )


class HarmonizationLSTM(nn.Module):
    """
    2-layer LSTM encoder on soprano; per-timestep heads predict alto, tenor, and bass.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.soprano_embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head_alto = nn.Linear(hidden_dim, vocab_size)
        self.head_tenor = nn.Linear(hidden_dim, vocab_size)
        self.head_bass = nn.Linear(hidden_dim, vocab_size)

    def forward(self, soprano: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.soprano_embed(soprano))
        hidden, _ = self.lstm(embedded)
        hidden = self.dropout(hidden)
        return (
            self.head_alto(hidden),
            self.head_tenor(hidden),
            self.head_bass(hidden),
        )


def harmonization_loss(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: nn.Module,
) -> torch.Tensor:
    """Average cross-entropy over alto, tenor, and bass predictions."""
    losses = []
    for logit, target in zip(logits, targets):
        losses.append(
            criterion(logit.reshape(-1, logit.size(-1)), target.reshape(-1))
        )
    return sum(losses) / len(losses)


def train_harmonization_epoch(
    model: HarmonizationLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    for sop, alto, tenor, bass in loader:
        sop = sop.to(device)
        targets = (alto.to(device), tenor.to(device), bass.to(device))
        optimizer.zero_grad()
        logits = model(sop)
        loss = harmonization_loss(logits, targets, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_harmonization(
    model: HarmonizationLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for sop, alto, tenor, bass in loader:
        sop = sop.to(device)
        targets = (alto.to(device), tenor.to(device), bass.to(device))
        logits = model(sop)
        loss = harmonization_loss(logits, targets, criterion)
        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


def train_harmonization_with_early_stopping(
    model: HarmonizationLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 8,
    checkpoint_path: str = "harmonization_best.pt",
) -> tuple[list[float], list[float], int]:
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    wait = 0
    best_epoch = 1

    for epoch in range(1, max_epochs + 1):
        train_loss = train_harmonization_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = evaluate_harmonization(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            marker = "  *" if val_loss < best_val else ""
            print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best val={best_val:.4f} at epoch {best_epoch})"
                )
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Loaded best weights from epoch {best_epoch} (val={best_val:.4f})")
    return train_losses, val_losses, best_epoch


def encode_chorales_for_split(
    encoded_chorales: Sequence[Sequence[Sequence[tuple]]],
    indices: Sequence[int],
    vocab: PitchDurationVocab,
) -> list[list[list[int]]]:
    """Encode selected chorales as four integer voice sequences."""
    result = []
    for idx in indices:
        voices = [vocab.encode(v) for v in encoded_chorales[idx]]
        result.append(voices)
    return result


def build_harmonization_dataloaders(
    encoded_chorales: Sequence[Sequence[Sequence[tuple]]],
    splits: ChoraleSplits,
    vocab: PitchDurationVocab,
    window_size: int = 32,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    split_chorales = {
        "train": encode_chorales_for_split(encoded_chorales, splits.train_indices, vocab),
        "val": encode_chorales_for_split(encoded_chorales, splits.val_indices, vocab),
        "test": encode_chorales_for_split(encoded_chorales, splits.test_indices, vocab),
    }
    loaders = {}
    for name, chorales in split_chorales.items():
        dataset = HarmonizationDataset(chorales, window_size=window_size)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
        )
    return loaders["train"], loaders["val"], loaders["test"]


@torch.no_grad()
def harmonize_soprano(
    model: HarmonizationLSTM,
    soprano_ids: Sequence[int],
    device: torch.device,
) -> tuple[list[int], list[int], list[int]]:
    """Predict alto, tenor, and bass token ids aligned to a soprano sequence."""
    model.eval()
    sop = torch.tensor([list(soprano_ids)], dtype=torch.long, device=device)
    logits_alto, logits_tenor, logits_bass = model(sop)
    alto = logits_alto.argmax(dim=-1).squeeze(0).tolist()
    tenor = logits_tenor.argmax(dim=-1).squeeze(0).tolist()
    bass = logits_bass.argmax(dim=-1).squeeze(0).tolist()
    return alto, tenor, bass


def harmonized_chorale_to_score(
    soprano_ids: Sequence[int],
    alto_ids: Sequence[int],
    tenor_ids: Sequence[int],
    bass_ids: Sequence[int],
    vocab: PitchDurationVocab,
    title: str = "",
) -> stream.Score:
    """Build a four-voice score from soprano + predicted lower voices."""
    voices = [soprano_ids, alto_ids, tenor_ids, bass_ids]
    score = stream.Score()
    for voice_name, tokens in zip(VOICE_NAMES, voices):
        part = tokens_to_part(tokens, vocab)
        part.partName = f"{title}_{voice_name}" if title else voice_name
        score.insert(part)
    return score


def combine_harmonizations_to_score(
    pieces: Sequence[tuple[str, list[int], list[int], list[int], list[int]]],
    vocab: PitchDurationVocab,
    gap_quarters: float = 4.0,
) -> stream.Score:
    """
    Lay out multiple harmonized chorales sequentially into 4 persistent parts.

    Each piece is (label, soprano, alto, tenor, bass).
    """
    import copy

    full_score = stream.Score()
    voice_parts = []
    for name in VOICE_NAMES:
        part = stream.Part()
        part.partName = name
        full_score.insert(0, part)
        voice_parts.append(part)

    offset = 0.0
    for label, sop, alto, tenor, bass in pieces:
        piece = harmonized_chorale_to_score(sop, alto, tenor, bass, vocab, title=label)
        piece_duration = 0.0
        for voice_part, piece_part in zip(voice_parts, piece.parts):
            for element in piece_part.flatten().notesAndRests:
                piece_duration = max(piece_duration, element.offset + element.quarterLength)
                el = copy.deepcopy(element)
                voice_part.insert(element.offset + offset, el)
        offset += piece_duration + gap_quarters

    return full_score


def harmonize_test_chorales(
    model: HarmonizationLSTM,
    encoded_chorales: Sequence[Sequence[Sequence[tuple]]],
    test_indices: Sequence[int],
    vocab: PitchDurationVocab,
    metadata: Sequence[dict],
    device: torch.device,
) -> list[tuple[str, list[int], list[int], list[int], list[int]]]:
    """Run inference on test chorales using ground-truth soprano melodies."""
    pieces = []
    for idx in test_indices:
        voices = [vocab.encode(v) for v in encoded_chorales[idx]]
        sop, alto_gt, tenor_gt, bass_gt = align_chorale_voices(voices)
        alto_pred, tenor_pred, bass_pred = harmonize_soprano(model, sop, device)
        label = f"RK{metadata[idx]['riemenschneider']}"
        pieces.append((label, sop, alto_pred, tenor_pred, bass_pred))
    return pieces
