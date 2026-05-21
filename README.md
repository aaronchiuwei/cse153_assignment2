# cse153_assignment2

JSB Chorale symbolic unconditioned generation (CSE 153 / 253 Assignment 2).

## Notebooks

1. `01_chorale_exploration.ipynb` — load corpus, EDA plots
2. `02_preprocessing_tokenization.ipynb` — vocabulary, 80/10/10 split, DataLoaders
3. `03_lstm_train_generate.ipynb` — 2-layer LSTM, training, generation, MIDI export
4. `04_harmonization.ipynb` — soprano-conditioned LSTM, test inference, `symbolic_conditioned.mid`

## Train & export (50 epochs, ~40 min on CPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_generate.py --epochs 50
```

Outputs: `symbolic_unconditioned.mid`, `chorale_lstm.pt`, `loss_curves.png`