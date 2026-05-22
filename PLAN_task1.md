# Task 1 Plan — Symbolic Unconditioned Generation

## What the assignment actually requires

The grader reads a single exported HTML notebook. It must be self-contained, clean, and readable without running code. Four sections are expected:

1. **Exploratory analysis** — dataset context, preprocessing steps, plots/tables
2. **Modeling** — problem formulation, model choice rationale, architecture walkthrough
3. **Evaluation** — what makes output "good", baselines, comparison
4. **Related work** — prior use of JSB chorales, prior LSTM/Markov music models, comparison to our results

---

## Current state (from notebooks 01–03)

### What exists and is good
- EDA: pitch distribution, duration distribution, voice ranges, piece lengths — all computed ✓
- Tokenization: (pitch, duration) vocabulary of 323 tokens, 80/10/10 split ✓
- Model: 2-layer LSTM, ~1M params, early stopping, val perplexity ~12.2 ✓
- Generation: 3 pieces at temperatures 0.7, 1.0, 1.3 ✓
- MIDI export ✓

### What is missing (required by rubric)

| Gap | Why it matters |
|-----|---------------|
| **Baselines** | Grader asks: "how do you show your model beats trivial methods?" |
| **Evaluation metrics beyond perplexity** | Perplexity ≠ musical quality. Need pitch/duration distribution overlap, interval stats |
| **Related work section** | Required section — 0 content exists |
| **Clean single notebook** | Four notebooks exist; grader shouldn't have to jump around |
| **Notebook needs re-run** | Current outputs are from a different machine (Aaron's paths); must be re-run on this machine |

---

## Plan: single notebook `task1_unconditioned.ipynb`

Merge notebooks 01, 02, 03, and add the missing pieces. Sections:

### Section 0 — Related Work (put first, as rubric suggests)
- JSB Chorales as benchmark: Boulanger-Lewandowski et al. 2012 (RNN-RBM), Allan & Williams 2004 (harmonisation), BachBot (Liang et al. 2017), DeepBach (Hadjeres et al. 2017)
- Our approach vs. prior: simpler LSTM LM, single voice at a time rather than all-four jointly
- Expected perplexity range from literature (BachBot: ~7, simple RNN: ~15–20)

### Section 1 — EDA
- Keep all existing plots from notebook 01
- Add: **transition heatmap** (top-30 most common pitch→pitch bigrams) — shows Markov structure in data
- Add: **interval histogram** (consecutive pitch differences in semitones) — shows step motion dominance
- Add: a summary statistics table (as Markdown, so it renders in HTML)

### Section 2 — Preprocessing & Tokenization
- Keep notebook 02 content
- Add: diagram/table showing the sliding-window construction

### Section 3 — Modeling
- Keep notebook 03 content (model definition, training loop, loss curves)
- Add: table comparing modeling approaches (Markov, RNN, Transformer) with pros/cons

### Section 4 — Evaluation

#### 4a. Baselines to implement
1. **Unigram sampler** — sample tokens i.i.d. from training token frequency. Trivially bad.
2. **Bigram (1st-order Markov) model** — next token sampled from empirical bigram distribution. Much better than unigram, still no long-range structure.
3. **LSTM (our model)** — the full model

#### 4b. Metrics to compute (on test set)
| Metric | Unigram | Bigram | LSTM |
|--------|---------|--------|------|
| Perplexity | high | medium | lowest |
| Pitch distribution KL divergence vs. train | low | low | low |
| Duration distribution KL divergence vs. train | low | low | low |
| Interval distribution match | bad | ok | best |
| % parallel 5ths/8ths | high | medium | lower |

- **KL divergence** of generated pitch/duration marginals vs. training marginals
- **Interval histogram** comparison (model-generated vs. real Bach)
- **Perplexity table** — the main number

#### 4c. Qualitative
- Play the MIDI at end of presentation
- Describe what sounds "Bach-like" vs. what sounds wrong

---

## What to build in the notebook

```
task1_unconditioned.ipynb
├── Section 0: Related Work (markdown cells)
├── Section 1: EDA
│   ├── [existing] pitch/duration/length/range plots
│   ├── [NEW] pitch bigram heatmap
│   └── [NEW] interval histogram
├── Section 2: Preprocessing
│   └── [existing] vocab, split, dataloader
├── Section 3: Modeling
│   ├── [existing] ChoraleLSTM definition + training
│   └── [NEW] model comparison table (markdown)
└── Section 4: Evaluation
    ├── [NEW] Unigram baseline class
    ├── [NEW] Bigram baseline class
    ├── [NEW] Perplexity table (Unigram / Bigram / LSTM)
    ├── [NEW] KL divergence: pitch & duration distributions
    ├── [NEW] Interval histogram: real vs. generated (per model)
    └── [existing] MIDI export
```

---

## Scope decisions

- **Do not retrain** — keep the same LSTM architecture. The model is fine for this assignment.
- **Do not add Transformer** — overkill, not asked for.
- **Bigram model** is the right baseline: it's non-trivial (captures local context), but clearly weaker than LSTM on interval/long-range structure. This is exactly the argument the rubric wants.
- **KL divergence** is computable without ground-truth generation alignment — just compare marginal histograms.
- Parallel 5ths/8ths check is nice-to-have but requires multi-voice alignment — skip for Task 1 (single-voice LM).

---

## Order of work

1. Create `task1_unconditioned.ipynb` skeleton with all sections
2. Port EDA content + add bigram heatmap + interval histogram
3. Port preprocessing + model sections
4. Implement Unigram and Bigram baseline classes
5. Implement perplexity evaluation for all three models
6. Implement KL divergence and interval histogram comparison
7. Write related work markdown cells
8. Re-run all cells clean, export to HTML
