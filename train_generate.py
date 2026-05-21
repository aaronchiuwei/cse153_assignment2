#!/usr/bin/env python3
"""Train the chorale LSTM and export symbolic_unconditioned.mid."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from chorale_data import (
    build_dataloaders,
    flatten_voice_sequences,
    load_chorales,
    split_chorale_indices,
    PitchDurationVocab,
)
from chorale_model import (
    ChoraleLSTM,
    combine_pieces_to_score,
    evaluate,
    export_midi,
    generate_piece,
    train_epoch,
)

ROOT = Path(__file__).resolve().parent
MIDI_PATH = ROOT / "symbolic_unconditioned.mid"
PLOT_PATH = ROOT / "loss_curves.png"
CHECKPOINT_PATH = ROOT / "chorale_lstm.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--voice-length", type=int, default=56)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Load checkpoint and generate MIDI without training",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoded_chorales, _ = load_chorales()
    splits = split_chorale_indices(len(encoded_chorales), seed=args.seed)
    train_sequences = flatten_voice_sequences(encoded_chorales, splits.train_indices)

    vocab = PitchDurationVocab()
    vocab.build_from_sequences(train_sequences)

    train_loader, val_loader, test_loader = build_dataloaders(
        encoded_chorales,
        splits,
        vocab,
        window_size=32,
        batch_size=args.batch_size,
    )

    model = ChoraleLSTM(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    test_loss = 0.0

    if args.generate_only and CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        test_loss = checkpoint.get("test_loss", 0.0)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print(f"Training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}",
                    flush=True,
                )

        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test loss: {test_loss:.4f}")

        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "test_loss": test_loss,
            },
            CHECKPOINT_PATH,
        )
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("LSTM language model — loss curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved loss plot to {PLOT_PATH}")

    encoded_train = [vocab.encode(seq) for seq in train_sequences]
    seed_chunks = [encoded_train[random.randrange(len(encoded_train))][:8] for _ in range(4)]

    temperatures = [0.7, 1.0, 1.3]
    pieces = []
    for temp in temperatures:
        voices = generate_piece(
            model,
            vocab,
            voices_length=args.voice_length,
            temperature=temp,
            seed_sequences=seed_chunks,
            device=device,
        )
        pieces.append((f"T{temp}", voices))
        print(f"Generated piece at temperature {temp}")

    score = combine_pieces_to_score(pieces, vocab)
    export_midi(score, str(MIDI_PATH))
    print(f"Exported MIDI to {MIDI_PATH}")


if __name__ == "__main__":
    main()
