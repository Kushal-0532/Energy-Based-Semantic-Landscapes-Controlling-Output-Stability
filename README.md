# Energy-Based Semantic Landscapes: Controlling Output Stability in Language Model Generation

Can a **learned energy function** make a small LLM give more consistent answers to the same question?
This project trains an energy-based model (EBM) on semantic-similarity data, refines a draft answer's BERT embedding with Langevin dynamics, and re-ranks the LM's candidate outputs by closeness to the refined vector. The LM stays frozen.

**Result in one line:** variance drops ~35%, but plain best-of-N re-ranking gets the same drop, so the EBM adds nothing measurable here (see [Results](#results)).

---

## Pipeline

```mermaid
flowchart LR
    subgraph DATA["Stations 1-2: Data"]
        A["STS-B + SNLI"] --> B["Training triples<br/>anchor / positive / negative"]
    end
    subgraph TRAIN["Station 3: EBM"]
        B --> C["Frozen BERT<br/>CLS embeddings"]
        C --> D["EnergyMLP<br/>NCE + margin loss"]
    end
    subgraph DECODE["Stations 4-5: Decode"]
        E["Prompt"] --> F["Draft from frozen LM"]
        F --> G["Langevin refinement<br/>in BERT space"]
        D -. energy .-> G
        G --> H["Re-rank LM candidates<br/>by cosine to refined vector"]
    end
    subgraph EVAL["Station 6: Eval"]
        H --> I["Semantic variance<br/>BERTScore F1"]
    end
```

### Stations 1-2: Data preparation

Each training example is a triple `(anchor, positive, negative)`. The positive is always a genuine paraphrase of the anchor, never the anchor itself (an identity positive makes `|anchor - positive| = 0` and the ranking task trivial).

```mermaid
flowchart TD
    S["STS-B<br/>sentence-transformers/stsb"] --> SC{"Score scale?"}
    SC -->|"max <= 1.0"| T1["pos >= 0.8<br/>neg <= 0.2"]
    SC -->|"max > 1.0"| T2["pos >= 4.0<br/>neg <= 1.0"]
    T1 --> P["Positives<br/>paraphrase pairs"]
    T2 --> P
    T1 --> EN["Easy negatives<br/>low-similarity pairs"]
    T2 --> EN
    P --> SH["Shuffled negatives<br/>word order permuted"]
    N["SNLI contradictions<br/>label = 2"] --> HN["Hard negatives"]
    P --> TA["Type A<br/>paraphrase + shuffled"]
    SH --> TA
    P --> TB["Type B<br/>paraphrase + SNLI contradiction"]
    HN --> TB
    P --> TC["Type C<br/>paraphrase + easy negative"]
    EN --> TC
    TA --> SPLIT["90% train / 10% held out"]
    TB --> SPLIT
    TC --> SPLIT
```

### Station 3: Energy model

A two-layer MLP (`EnergyMLP`, 197,121 params) scores `|anchor - candidate|` (768-d, from frozen `bert-base-uncased` CLS embeddings). Trained with noise-contrastive estimation and `MarginRankingLoss` so that coherent pairs get **lower** energy than incoherent ones. The held-out split reports ranking accuracy alongside the loss curve.

```mermaid
flowchart LR
    A["anchor"] --> BA["BERT CLS"]
    P["positive"] --> BP["BERT CLS"]
    N["negative"] --> BN["BERT CLS"]
    BA --> DP["abs diff"]
    BP --> DP
    BA --> DN["abs diff"]
    BN --> DN
    DP --> EP["EnergyMLP"] --> EPV["E positive"]
    DN --> EN["EnergyMLP<br/>shared weights"] --> ENV["E negative"]
    EPV --> L["MarginRankingLoss<br/>want E pos + margin < E neg"]
    ENV --> L
```

### Stations 4-5: Langevin decoding with drift penalty

```mermaid
sequenceDiagram
    participant U as Prompt
    participant LM as Frozen LM
    participant B as BERT
    participant L as Langevin loop
    participant R as Re-ranker
    U->>B: encode prompt (reference vector)
    U->>LM: sample draft
    LM->>B: encode draft (start vector)
    B->>L: start vector
    loop n_steps
        Note over L: E_total = EBM energy + alpha * (1 - cos to prompt)
        L->>L: gradient step + Gaussian noise
    end
    U->>LM: sample n_candidates
    LM->>R: candidate texts
    L->>R: refined vector
    R->>R: rank candidates by cosine to refined vector
    R-->>U: best candidate
```

### Station 6: Evaluation

10 prompts x 5 runs under three arms. The rerank-only arm isolates the EBM's contribution from plain best-of-N re-ranking.

```mermaid
flowchart LR
    PR["10 prompts x 5 runs"] --> A1["Baseline<br/>raw LM sampling"]
    PR --> A2["Rerank-only<br/>candidates ranked vs unrefined draft"]
    PR --> A3["EBM-guided<br/>Langevin + rerank"]
    A1 --> M["Semantic variance<br/>1 - mean pairwise cosine, lower is better"]
    A2 --> M
    A3 --> M
    A1 --> BS["BERTScore F1 vs prompt<br/>higher is better"]
    A2 --> BS
    A3 --> BS
    A3 -. "EBM share = EBM minus rerank-only" .-> A2
```

---

## Results

Latest notebook run: scale-aware STS-B threshold, non-identical positives, 10% held out. The language model was the **TinyLlama-1.1B-Chat fallback** (Phi-3-mini failed to load).

**Data:** 1,406 STS-B positives, 1,103 easy negatives, 3,000 SNLI contradictions -> 3,915 triples (3,523 train / 392 held out).

**EBM training:** loss 0.35 -> 0.13 over 5 epochs. Held-out ranking accuracy **365/392 = 93.1%**, mean energy margin (neg - pos) 2.25.

| System | Mean semantic variance | Reduction vs baseline | BERTScore F1 |
|---|---|---|---|
| Baseline (raw sampling) | 0.1458 | - | 0.8536 |
| Rerank-only (no EBM) | 0.0941 | 35.5% | 0.8532 |
| EBM (Langevin + rerank) | 0.0954 | 34.6% | 0.8534 |

```mermaid
xychart-beta
    title "Mean semantic variance (lower = more consistent)"
    x-axis ["Baseline", "Rerank-only", "EBM"]
    y-axis "Variance" 0 --> 0.16
    bar [0.1458, 0.0941, 0.0954]
```

**Takeaway:** the EBM pipeline cuts output variance by ~35% with no BERTScore loss, but plain best-of-N re-ranking gets the same reduction (EBM minus rerank-only = -0.9 pts; EBM wins on 5 of 10 prompts, loses on 5). The EBM scores well on its own ranking task, yet Langevin refinement adds no measurable stability beyond re-ranking. With 10 prompts x 5 runs and a single seed, differences this small are within noise.

---

## Comparison with COLD Decoding

Inspired by COLD Decoding (Qin et al., NeurIPS 2022). Both freeze the LM, apply Langevin dynamics in a continuous space, and inject noise during optimization.

| Aspect | COLD Decoding | This work |
|---|---|---|
| Energy function | Hand-crafted constraints | Learned MLP trained on STS-B + SNLI |
| Optimization space | Token logit space | BERT embedding space (768-d) |
| Decoding method | Soft tokens projected to vocabulary | Re-ranking from LM candidate pool |
| Drift control | LM log-probability | Cosine similarity to prompt embedding |
| Primary objective | Constraint satisfaction | Output variance reduction |
| Training required | No | Yes (NCE on semantic similarity data) |

---

## Repository structure

```mermaid
flowchart LR
    ROOT["repo"] --> DP["dataset_prep.py<br/>builds data files"]
    ROOT --> NB["ebm_stability/<br/>ebm_stability_pipeline-final.ipynb"]
    ROOT --> DATA["data/"]
    ROOT --> CK["checkpoints/<br/>created at runtime"]
    ROOT --> RES["results/<br/>created at runtime"]
    DATA --> D1["positives.json"]
    DATA --> D2["negatives_easy.json"]
    DATA --> D3["negatives_hard.json"]
    DATA --> D4["negatives_shuffled.json"]
    CK --> C1["energy_mlp.pt"]
    RES --> R1["variance_table.csv"]
    RES --> R2["variance_comparison.png"]
    RES --> R3["training_loss.png"]
```

---

## Setup and usage

```bash
pip install transformers datasets torch scikit-learn bert-score tqdm numpy pandas matplotlib accelerate bitsandbytes
```

Open `ebm_stability/ebm_stability_pipeline-final.ipynb` and run cells in order. It handles data loading, EBM training, Langevin decoding, and evaluation end to end. To skip retraining, load `checkpoints/energy_mlp.pt` before the evaluation cells.

**Hardware:** a CUDA GPU is recommended. The LM is loaded in 4-bit (bitsandbytes) when available; CPU works but is much slower.

**Reproducibility:** global seed 42 for Python, NumPy, and PyTorch.

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | 10 | Langevin refinement iterations |
| `alpha` | 0.5 | Weight of drift penalty relative to EBM energy |
| `step_size` | 0.1 | Gradient step magnitude in Langevin update |
| `noise_scale` | 0.01 | Gaussian noise injected per step |
| `n_candidates` | 8 | LM samples for re-ranking |
| `temperature` | 0.9 | LM sampling temperature |
| `NUM_EPOCHS` | 5 | EBM training epochs |
| `margin` | 1.0 | MarginRankingLoss margin |
| `learning_rate` | 2e-4 | Adam learning rate |
| `batch_size` | 32 | Training batch size |

## Models and datasets

- **BERT:** `bert-base-uncased` (frozen encoder)
- **LM:** `microsoft/Phi-3-mini-4k-instruct`, fallback `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (used in the latest run)
- **Data:** `sentence-transformers/stsb`, `stanfordnlp/snli`
