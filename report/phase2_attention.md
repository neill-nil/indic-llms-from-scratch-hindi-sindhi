# Phase 2: Attention Analysis

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.eval.attention
       python -m common.eval.plot_attention
       python -m common.eval.write_attention_report
-->

What the multi-head attention actually learned, measured on 64 held-out
windows of 256 tokens per model, drawn from the validation split and screened to
contain no document separator.

## Method

Three statistics per (layer, head):

* **Normalised entropy** — Shannon entropy of the attention distribution divided by
  `log2(t+1)`, the maximum available at query position *t*. **1.0** means uniform over
  everything visible; **0.0** means all mass on a single key. Normalising matters here
  because the two models have different context lengths, so raw entropy would not be
  comparable between them.
* **Mean attention distance** — `sum_j p(t,j) * (t - j)`, in tokens. How far back a head
  typically looks.
* **Mass on position *t-1*** — isolates previous-token heads.

Query positions below **32** are excluded from every aggregate. Under a
causal mask, query *t* can only see *t+1* keys, so early queries are *forced* to be
low-entropy and short-range. Including them would confound "this head is local" with "this
head had nowhere else to look", and would bias the two models differently.

## Figures

| File | Content |
|---|---|
| `figures/attention_hindi_layer0.png` | Model H, early layer, 4 heads, real Devanagari tokens |
| `figures/attention_hindi_layer5.png` | Model H, late layer |
| `figures/attention_sindhi_layer0.png` | Model L, early layer, real Perso-Arabic tokens |
| `figures/attention_sindhi_layer5.png` | Model L, late layer |
| `figures/attention_head_statistics.png` | Entropy and distance for every head, both models |
| `figures/attention_depth_trends.png` | Layer-wise means, models overlaid |

The heatmaps are colour-clipped at the 99th percentile. Without that, the position-0
column saturates the scale and the rest of the matrix renders uniformly dark — see the
attention-sink section below.

## Normalised entropy per head

**Model H (Hindi)**

| Layer | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| **0** | 0.97 | 0.97 | 0.98 | 0.96 | 0.94 | 0.86 | 0.93 | 0.96 | 0.95 |
| **1** | 0.89 | 0.88 | 0.95 | 0.85 | 0.66 | 0.63 | 0.89 | 0.85 | 0.83 |
| **2** | 0.36 | 0.81 | 0.53 | 0.44 | 0.38 | 0.75 | 0.56 | 0.50 | 0.54 |
| **3** | 0.38 | 0.32 | 0.50 | 0.38 | 0.41 | 0.86 | 0.64 | 0.03 | 0.44 |
| **4** | 0.68 | 0.45 | 0.65 | 0.44 | 0.38 | 0.09 | 0.49 | 0.46 | 0.46 |
| **5** | 0.44 | 0.71 | 0.80 | 0.64 | 0.64 | 0.73 | 0.55 | 0.66 | 0.65 |

**Model L (Sindhi)**

| Layer | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| **0** | 0.91 | 0.97 | 0.96 | 0.96 | 0.94 | 0.88 | 0.96 | 0.95 | 0.94 |
| **1** | 0.84 | 0.60 | 0.80 | 0.97 | 0.45 | 0.77 | 0.66 | 0.81 | 0.74 |
| **2** | 0.58 | 0.79 | 0.87 | 0.61 | 0.85 | 0.49 | 0.48 | 0.28 | 0.62 |
| **3** | 0.26 | 0.46 | 0.44 | 0.52 | 0.59 | 0.37 | 0.55 | 0.39 | 0.45 |
| **4** | 0.57 | 0.74 | 0.77 | 0.39 | 0.02 | 0.40 | 0.34 | 0.63 | 0.48 |
| **5** | 0.50 | 0.73 | 0.28 | 0.83 | 0.49 | 0.49 | 0.74 | 0.70 | 0.60 |

## Mean attention distance per head (tokens)

**Model H (Hindi)**

| Layer | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| **0** | 78 | 74 | 70 | 83 | 64 | 40 | 53 | 75 | 67 |
| **1** | 88 | 99 | 56 | 37 | 14 | 40 | 75 | 40 | 56 |
| **2** | 5 | 27 | 16 | 6 | 14 | 93 | 42 | 10 | 27 |
| **3** | 10 | 3 | 10 | 5 | 8 | 57 | 18 | 1 | 14 |
| **4** | 29 | 12 | 36 | 19 | 8 | 8 | 12 | 87 | 27 |
| **5** | 103 | 71 | 58 | 55 | 78 | 62 | 91 | 73 | 74 |

**Model L (Sindhi)**

| Layer | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| **0** | 60 | 71 | 71 | 65 | 70 | 48 | 84 | 68 | 67 |
| **1** | 68 | 105 | 38 | 73 | 119 | 27 | 82 | 42 | 69 |
| **2** | 11 | 70 | 94 | 19 | 36 | 14 | 79 | 116 | 55 |
| **3** | 3 | 5 | 5 | 11 | 15 | 5 | 11 | 4 | 7 |
| **4** | 15 | 35 | 65 | 5 | 1 | 8 | 11 | 24 | 20 |
| **5** | 93 | 71 | 112 | 54 | 84 | 87 | 71 | 50 | 78 |

## What the heads are doing

### Both models learned the same three-phase structure with depth

This is the clearest result, and it holds independently in two models trained on different
languages, different corpora and different hardware:

| Phase | Layers | Entropy (H / L) | Distance (H / L) | Behaviour |
|---|---|---|---|---|
| Broad survey | 0–1 | 0.95 / 0.94 | 67 / 67 | near-uniform mixing over the window |
| Specialisation | 2–4 | 0.48 / 0.52 | 22 / 28 | sharp, short-range, positional |
| Re-integration | 5 | 0.65 / 0.60 | 74 / 78 | broad again, long-range |

Early layers average widely, middle layers commit to specific nearby positions, and the
final layer re-broadens to gather context for the prediction. Nothing in the architecture
or the loss asks for this arrangement — it emerged from training, twice.

### Named heads — Model H (Hindi)

* **L3H7 — previous-token head.** Entropy 0.03, mean distance 1.1 tokens, with 94% of its mass on position *t-1*. It reads the immediately preceding token and nothing else.
* **L5H0 — longest-range head.** Mean distance 103 tokens at entropy 0.44: it integrates across the window rather than tracking a fixed offset.
* **L0H2 — most diffuse head.** Entropy 0.98 of a possible 1.0, i.e. close to uniform over everything visible; it behaves more like an averaging operation than a lookup.

### Named heads — Model L (Sindhi)

* **L4H4 — previous-token head.** Entropy 0.02, mean distance 1.1 tokens, with 96% of its mass on position *t-1*. It reads the immediately preceding token and nothing else.
* **L1H4 — longest-range head.** Mean distance 119 tokens at entropy 0.45: it integrates across the window rather than tracking a fixed offset.
* **L0H1 — most diffuse head.** Entropy 0.97 of a possible 1.0, i.e. close to uniform over everything visible; it behaves more like an averaging operation than a lookup.

Both models independently converged on a near-pure **previous-token head**: entropy below
0.05 and mean distance of essentially exactly 1. That is the classic positional primitive
— it does not read content at all, only offset — and it is a known component of induction
circuits. Finding it in two separately-trained models is evidence that it is a useful
solution rather than an accident of one run.

### Positional versus content-based heads

The two categories separate cleanly on the measurements. **Positional** heads sit at low
entropy and short, near-integer distance: they attend by offset regardless of what the
tokens are, and they cluster in layers 3–4 of both models. **Content-based** heads sit at
moderate-to-high entropy with long and variable distance — in the heatmaps their bright
cells scatter irregularly rather than forming a clean band, because where they look
depends on what the text says. Layer 5 is dominated by these in both models.

### The attention sink

A large share of all attention mass lands on **position 0**, regardless of what that token
is:

| | mass on position 0 | mass on everything else |
|---|---|---|
| Model H (Hindi) | 16.9% | 83.1% |
| Model L (Sindhi) | 23.8% | 76.2% |

This is the well-documented *attention sink*. Softmax forces every row to sum to 1, so a
head with nothing relevant to attend to must still put its mass somewhere; the first
position is a stable, always-available default. It is the reason the heatmaps need
percentile clipping to be readable at all.

**Model L parks 1.4x more mass on the sink than Model H**
(23.8% vs 16.9%). One reading is that the Sindhi model more
often finds nothing useful to attend to and falls back on the default — consistent with its
higher bits-per-byte. This is suggestive rather than established: it is measured on the
example windows and would need a controlled test to claim as cause.

## How the two models differ

They are more alike than different, which is itself worth stating: identical architecture,
identical token budget and identical optimisation produced the same depth structure, the
same head types, and the same previous-token primitive.

The differences that do appear:

1. **Model L relies more on the attention sink** (23.8% vs
   16.9%), suggesting more positions where no informative context is found.
2. **Model L's middle layers are slightly more diffuse**
   (layers 2–4 mean entropy 0.52 vs 0.48), i.e. its
   specialised heads commit less sharply.
3. **Model L's final layer reaches further**
   (78 vs 74 tokens), which is expected given its
   1024-token context against Model H's 512 — it has more window to reach across.

Point 3 is a direct consequence of the context-length decision recorded in
`context_length_decision.md`, not an independent finding.

Raw measurements: `report/attention_results.json`.
