# Energy-Based Semantic Landscapes: Controlling Output Stability in Language Model Generation

## Overview

This project looks at the use of Energy-Based Models (EBMs) combined with Langevin dynamics in hopes to reduce output variance in LLM generation. Rather than using hand crafting constraints, we train a learned energy function on semantic similarity data and operate in BERT embedding space to guide generation toward more stable and semantically coherent outputs across repeated runs.

The core idea is to refine a draft generation by performing gradient based optimization in the embedding space, penalizing both semantic incoherence (via the learned EBM) and drift from the original prompt (via a cosine similarity penalty). Candidates from the language model are then re-ranked by closeness to the refined vector.


---

## Method

The pipeline is organized into six stations:

**Stations 1-2: Data Preparation**

Training data is constructed from two sources:
- Positive pairs: STS-B sentence pairs above a similarity threshold
- Hard negatives: contradiction pairs from SNLI (label = 2)
- Easy negatives: STS-B pairs below a low similarity threshold
- Shuffled negatives: positive pairs with word order randomly permuted

The thresholds are picked from the data: `sentence-transformers/stsb` normalises scores to `[0, 1]`, while
the original STS-B release uses `[0, 5]`, so the code reads the scale and uses `>= 0.8 / <= 0.2` or
`>= 4.0 / <= 1.0` accordingly. Three types of training triples are formed and saved to disk. In every
triple the positive is a genuine paraphrase of the anchor, never the anchor itself — an identity positive
makes `|anchor - positive| = 0` and the ranking task trivial.

**Station 3: Energy Model Training**

A two-layer MLP (EnergyMLP) is trained on top of frozen BERT CLS embeddings. The input is the absolute difference between anchor and candidate vectors (768-dimensional). Training uses Noise Contrastive Estimation with MarginRankingLoss (margin=1.0) and an Adam optimizer (lr=2e-4) over 5 epochs. The goal is to assign lower energy to semantically coherent pairs and higher energy to incoherent ones. 10% of the triples are held out, and the notebook reports ranking accuracy (the fraction of unseen triples where the positive scores lower than the negative) alongside the loss curve.

**Stations 4-5: Langevin Decoding with Drift Penalty**

Given a prompt:
1. Encode the prompt with BERT to obtain a reference embedding.
2. Sample an initial draft from the language model and encode it.
3. Run Langevin refinement for `n_steps` iterations:
   - Compute total energy = EBM energy + alpha * (1 - cosine_similarity to prompt)
   - Update the embedding via gradient descent plus Gaussian noise.
4. Generate `n_candidates` outputs from the language model and re-rank them by cosine similarity to the refined embedding.

The language model (Phi-3-mini-4k-instruct, with TinyLlama-1.1B-Chat as fallback) is kept fully frozen throughout.

**Station 6: Evaluation**

Ten semantic prompts are each run five times under three conditions: baseline (raw LM sampling), rerank-only
(the same candidate pool ranked against the unrefined draft embedding — no EBM, no Langevin) and EBM-guided.
The rerank-only arm is the ablation that separates the EBM's contribution from plain best-of-N re-ranking.
Metrics:
- Semantic variance: 1 - mean pairwise cosine similarity across outputs (lower is more consistent)
- BERTScore F1: similarity between outputs and the prompt (higher is more relevant)

---

## Comparison with COLD Decoding

This work is directly inspired by COLD Decoding (Qin et al., NeurIPS 2022) but differs in several key ways:

| Aspect | COLD Decoding | This Work |
|---|---|---|
| Energy function | Hand-crafted constraints | Learned MLP trained on STS-B + SNLI |
| Optimization space | Token logit space | BERT embedding space (768d) |
| Decoding method | Soft tokens projected to vocabulary | Re-ranking from LM candidate pool | 
| Drift control | LM log-probability | Cosine similarity to prompt embedding |
| Primary objective | Constraint satisfaction | Output variance reduction |
| Training required | No | Yes (NCE on semantic similarity data) |

Both methods freeze the language model, apply Langevin dynamics in a continuous space, and inject stochastic noise during optimization.

---

## Repository Structure

```
.
├── dataset_prep.py                        # Standalone script for building data files
├── ebm_stability/
│   └── ebm_stability_pipeline-final.ipynb       # Full end-to-end pipeline notebook
├── data/
│   ├── positives.json
│   ├── negatives_easy.json
│   ├── negatives_hard.json
│   └── negatives_shuffled.json
├── checkpoints/                           # Created at runtime
│   └── energy_mlp.pt
└── results/                               # Created at runtime
    ├── variance_table.csv
    ├── variance_comparison.png
    └── training_loss.png
```

---

## Setup and Usage

**Install dependencies:**

```bash
pip install transformers datasets torch scikit-learn bert-score tqdm numpy pandas matplotlib accelerate bitsandbytes
```

**Run the pipeline:**

Open `ebm_stability/ebm_stability_pipeline-final.ipynb` and run cells sequentially. The notebook handles dataset loading, model training, Langevin decoding, and evaluation end-to-end.

To skip retraining, load a saved checkpoint from `checkpoints/energy_mlp.pt` before running the evaluation cells.

**Hardware:** A CUDA-compatible GPU is recommended. 4-bit quantization (via bitsandbytes) is used for the language model when available. CPU fallback is supported but will be significantly slower.

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | 10 | Langevin refinement iterations |
| `alpha` | 0.5 | Weight of drift penalty relative to EBM energy |
| `step_size` | 0.1 | Gradient step magnitude in Langevin update |
| `noise_scale` | 0.01 | Gaussian noise injected per step |
| `n_candidates` | 8 | Number of LM samples for re-ranking |
| `temperature` | 0.9 | LM sampling temperature |
| `NUM_EPOCHS` | 5 | EBM training epochs |
| `margin` | 1.0 | MarginRankingLoss margin |
| `learning_rate` | 2e-4 | Adam optimizer learning rate |
| `batch_size` | 32 | Training batch size |

---

## Models and Datasets

**Pre-trained models:**
- BERT: `bert-base-uncased` (frozen encoder)
- Language model: `microsoft/Phi-3-mini-4k-instruct` (fallback: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

**Datasets:**
- STS-B: `sentence-transformers/stsb`
- SNLI: `snli`

---

## Results (latest notebook run)

Fixed run (scale-aware STS-B threshold, non-identical positives, 10% held out, Phi-3-mini loaded).

**Data:** 1406 STS-B positives, 1103 easy negatives, 3000 SNLI contradictions -> 3915 triples (3523 train / 392 held out).

**EBM training:** loss 0.35 -> 0.13 over 5 epochs (no longer collapses to ~0). Held-out ranking accuracy **365/392 = 93.1%**, mean energy margin (neg - pos) 2.25.

**Evaluation (10 prompts x 5 runs):**

| System | Mean semantic variance | Reduction vs baseline | BERTScore F1 |
|---|---|---|---|
| Baseline (raw sampling) | 0.1458 | - | 0.8536 |
| Rerank-only (no EBM) | 0.0941 | 35.5% | 0.8532 |
| EBM (Langevin + rerank) | 0.0954 | 34.6% | 0.8534 |

**Takeaway:** the EBM pipeline cuts output variance by ~35% with no BERTScore loss, but plain best-of-N re-ranking gets the same reduction (EBM minus rerank-only = -0.9 pts). The EBM scores well on its ranking task, yet Langevin refinement adds no measurable stability beyond re-ranking here. With 10 prompts x 5 runs and a single seed, differences this small are within noise.

## Reproducibility

A global seed of 42 is set for Python, NumPy, and PyTorch at the start of the pipeline.
