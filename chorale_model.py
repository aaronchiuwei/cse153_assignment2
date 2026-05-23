"""2-layer LSTM language model for JSB chorale token sequences."""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn as nn
from music21 import note, stream
from torch.utils.data import DataLoader

from chorale_data import PAD_TOKEN, UNK_TOKEN, PitchDurationVocab, VOICE_NAMES

SPECIAL_IDS = {0, 1}  # PAD, UNK


class ChoraleLSTM(nn.Module):
    """Next-token LSTM language model over discrete (pitch, duration) tokens."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,  # increased from 0.2 to reduce overfitting
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        logits = self.output(self.dropout(output))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
        )


def train_epoch(
    model: ChoraleLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(
    model: ChoraleLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    return total_loss / total_tokens


def train_with_early_stopping(
    model: ChoraleLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 8,
    checkpoint_path: str = "best_model.pt",
) -> tuple[list[float], list[float], int]:
    """
    Train with early stopping: saves the best model weights when val loss
    improves, stops when val loss has not improved for `patience` epochs.

    Returns (train_losses, val_losses, best_epoch).
    """
    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val = float("inf")
    wait = 0
    best_epoch = 1

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}"
                  + ("  *" if val_loss < best_val else ""))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val={best_val:.4f} at epoch {best_epoch})")
                break

    # restore best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Loaded best weights from epoch {best_epoch} (val={best_val:.4f})")
    return train_losses, val_losses, best_epoch


def _sample_next(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(logits.argmax().item())
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1).item())


# Standard SATB voice pitch ranges (MIDI)
VOICE_PITCH_RANGES: dict[str, tuple[int, int]] = {
    "Soprano": (60, 81),  # C4–A5
    "Alto":    (53, 74),  # F3–D5
    "Tenor":   (48, 69),  # C3–A4
    "Bass":    (36, 62),  # C2–D4
}


@torch.no_grad()
def generate_sequence(
    model: ChoraleLSTM,
    vocab: PitchDurationVocab,
    length: int,
    temperature: float = 1.0,
    seed_ids: Sequence[int] | None = None,
    context_size: int = 32,
    pitch_range: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> list[int]:
    """Autoregressively generate a token sequence of the given length."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    valid_ids = [i for i in range(len(vocab)) if i not in SPECIAL_IDS]

    # Build mask: allow tokens whose pitch is in range (or is a rest)
    if pitch_range is not None:
        lo, hi = pitch_range
        mask = torch.full((len(vocab),), float("-inf"))
        for tid in valid_ids:
            pitch, _ = vocab.id_to_token[tid]
            if pitch is None or (lo <= pitch <= hi):  # rest or in-range note
                mask[tid] = 0.0
        # Fall back to all valid if mask is empty (shouldn't happen)
        if mask[valid_ids].max() == float("-inf"):
            mask = torch.zeros(len(vocab))
        mask = mask.to(device)
    else:
        mask = None

    if seed_ids:
        generated = list(seed_ids)
    else:
        start_pool = valid_ids if mask is None else [
            tid for tid in valid_ids if mask[tid] == 0.0
        ]
        generated = [random.choice(start_pool)]

    hidden = None
    while len(generated) < length:
        x = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)

        logits, hidden = model(x, hidden)
        lg = logits[0, -1]
        if mask is not None:
            lg = lg + mask
        next_id = _sample_next(lg, temperature)
        if next_id in SPECIAL_IDS:
            fallback = valid_ids if mask is None else [tid for tid in valid_ids if mask[tid] == 0.0]
            next_id = random.choice(fallback)
        generated.append(next_id)

    return generated[:length]


def tokens_to_part(token_ids: Sequence[int], vocab: PitchDurationVocab) -> stream.Part:
    """Decode integer tokens into a music21 Part."""
    part = stream.Part()

    for token_id in token_ids:
        if token_id in SPECIAL_IDS:
            continue
        pitch, duration = vocab.id_to_token[token_id]
        if pitch is None:
            part.append(note.Rest(quarterLength=duration))
        else:
            part.append(note.Note(midi=pitch, quarterLength=duration))
    return part


def voices_to_score(
    voice_token_lists: Sequence[Sequence[int]],
    vocab: PitchDurationVocab,
    piece_label: str = "",
) -> stream.Score:
    """Build a four-voice SATB Score from four generated token sequences."""
    score = stream.Score()
    for voice_name, tokens in zip(VOICE_NAMES, voice_token_lists):
        part = tokens_to_part(tokens, vocab)
        label = f"{piece_label}_{voice_name}" if piece_label else voice_name
        part.partName = label
        score.insert(part)
    return score


def combine_pieces_to_score(
    pieces: Sequence[tuple[str, list[list[int]]]],
    vocab: PitchDurationVocab,
    gap_quarters: float = 4.0,
) -> stream.Score:
    """Lay out multiple four-voice pieces sequentially in one Score."""
    full_score = stream.Score()
    offset = 0.0

    for label, voices in pieces:
        piece_score = voices_to_score(voices, vocab, piece_label=label)
        piece_duration = 0.0
        for part in piece_score.parts:
            for element in part.flatten().notesAndRests:
                element.offset += offset
                piece_duration = max(piece_duration, element.offset + element.quarterLength)
            full_score.insert(part)
        offset += piece_duration + gap_quarters

    return full_score


def generate_piece(
    model: ChoraleLSTM,
    vocab: PitchDurationVocab,
    voices_length: int,
    temperature: float,
    seed_sequences: Sequence[Sequence[int]] | None = None,
    device: torch.device | None = None,
) -> list[list[int]]:
    """Generate one four-voice piece (unconditioned across voices)."""
    voices = []
    for i, voice_name in enumerate(VOICE_NAMES):
        seed = list(seed_sequences[i][:8]) if seed_sequences and i < len(seed_sequences) else None
        pitch_range = VOICE_PITCH_RANGES.get(voice_name)
        voices.append(
            generate_sequence(
                model,
                vocab,
                length=voices_length,
                temperature=temperature,
                seed_ids=seed,
                pitch_range=pitch_range,
                device=device,
            )
        )
    return voices


def export_midi(score: stream.Score, path: str) -> None:
    score.write("midi", fp=path)