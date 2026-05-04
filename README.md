# Predicting FOMC Rate Decisions with Fine-Tuned FinBERT

Fine-tuning a transformer language model on 30 years of Federal Open Market Committee meeting minutes to predict the next rate decision (hike, hold, cut), then diagnosing why the model fails on the 2022–2026 tightening cycle.

This was my coursework for Advanced Machine Learning in Finance at UCL (MSc Computational Finance, 2025–2026). Final grade: 79.5% (Distinction).

📄 **[Full report (PDF)](./report.pdf)** &nbsp;·&nbsp; 📓 **[Notebook](./full_code.ipynb)**

---

## TL;DR

- Fine-tuned **FinBERT (110M parameters)** on **277 FOMC meetings (1993–2026)** for three-class rate-decision prediction (hike / hold / cut).
- Built a **TF-IDF + Logistic Regression baseline** to isolate the contribution of contextual modelling.
- Used **chronological splits** (no random shuffling) to prevent lookahead, **weighted cross-entropy** for class imbalance, **grid-searched hyperparameters** by validation macro F1, and **majority-vote aggregation** from sentences to meetings.
- **Validation sentence-level macro F1: 0.41** — meaningful signal above random baseline 0.33.
- **Test meeting-level macro F1: 0.19** (FinBERT) and **0.26** (baseline) — both fail to generalise.
- **Headline finding: signal exists at sentence level but does not survive aggregation under regime shift.** The 2022–2026 test period is the most aggressive tightening cycle in decades — a regime barely represented in training data — and FinBERT collapses to predicting "hold" for all 32 test meetings.

The negative result is the interesting finding: it shows the gap between in-distribution learning and out-of-distribution generalisation, which is exactly the failure mode that matters for any model deployed in financial markets.

## Methods at a glance

| Component       | Choice                                                            | Why                                                                      |
|-----------------|-------------------------------------------------------------------|--------------------------------------------------------------------------|
| Model           | FinBERT (yiyanghkust/finbert-tone)                                | Pre-trained on financial text — better domain priors than vanilla BERT   |
| Baseline        | TF-IDF + multinomial Logistic Regression                          | Cleanly attributes performance to contextual modelling vs lexical signal |
| Splits          | Chronological (train: 1993–2017, val: 2018–2021, test: 2022–2026) | No lookahead; reflects deployment reality                                |
| Loss            | Class-weighted cross-entropy                                      | Hold-class dominance would otherwise drown out hike/cut learning         |
| Hyperparameters | Grid search over learning rate × batch size, selected on val F1   | Optimal: lr=1e-5, batch=16, 3 epochs                                     |
| Aggregation     | Sentence → meeting via majority vote                              | Each meeting → ~50 sentences; meeting label is the prediction target     |
| Evaluation      | Macro F1, per-class F1, confusion matrix on held-out test set     | Class-imbalanced; macro F1 prevents inflation by majority class          |

## Results

| Metric (held-out test set, 2022–2026)   | FinBERT | TF-IDF + LR | Random |
|-----------------------------------------|---------|-------------|--------|
| Meeting-level accuracy                  | 0.41    | 0.45        | —      |
| Meeting-level macro F1                  | 0.19    | 0.26        | 0.33   |
| Hold F1                                 | 0.58    | 0.00        | —      |
| Hike F1                                 | 0.00    | 0.17        | —      |
| Cut F1                                  | 0.00    | 0.60        | —      |

Validation sentence-level macro F1 was **0.41** for FinBERT (above random baseline 0.33), confirming the model learned genuine policy signal — but this signal does not survive majority-vote aggregation under the 2022–2026 distribution shift.

## What I'd do differently

- **Hierarchical attention** at the meeting level instead of majority voting — sentences carrying the rate signal are a small fraction of any meeting, and majority voting drowns them out.
- **Walk-forward retraining** on a rolling window rather than a single chronological split — the regime-shift failure suggests the model's priors go stale fast.
- **Combine with macro indicators** — text alone is unlikely to be sufficient; pairing language signal with quantitative macro features would test whether language is incremental or substitutive.
- **Calibration analysis** on predicted probabilities — for any trading application, calibrated probabilities of a hike matter more than a thresholded prediction.
- **Cross-bank generalisation** — extending to ECB, BoE, and BoJ minutes would test whether the failure mode is FOMC-specific or general to central bank language.

## Reproducing this

> **Note on runtime:** the FinBERT fine-tuning takes ~2–3 hours on a free Colab T4 GPU. The notebook is shipped with outputs saved so you can read the results without re-running. The first few cells handle data download (FOMC minutes scraped from the Federal Reserve site, rate decisions via the FRED API).

```bash
