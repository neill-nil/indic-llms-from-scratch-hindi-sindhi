"""
Model configuration for the decoder-only Transformer language models.

One configuration object per model. Model H (Hindi) and Model L (Sindhi) use the same
architecture but are entirely separate runs: separate corpora, separate tokenizers,
separate vocabularies, separate weights. Nothing is shared between them at train or
inference time.
"""

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


@dataclass
class GPTConfig:
    """
    Hyperparameters for one decoder-only Transformer LM.

    The parameter budget is ~25M. With a tied input/output embedding the breakdown at the
    default settings is:

        token embedding   16,000 x 512  =  8,192,000   (33.8%)
        position embedding   512 x 512  =    262,144   ( 1.1%)
        6 transformer blocks            = 15,765,504   (65.1%)
        final LayerNorm                 =      1,024
        ------------------------------------------------------
        total                           = 24,220,672

    Why this shape. Three configurations land within 0.5% of each other at ~24.2M:
    (d=512, L=6, ff=1536), (d=448, L=7, ff=1792) and (d=512, L=5, ff=2048). We take the
    first because depth buys more than feed-forward width at this scale, so 6 layers beat
    5, and because d=512 with 8 heads keeps d_k at 64 with power-of-two dimensions that
    map cleanly onto tensor cores. The cost is a feed-forward ratio of 3x rather than the
    conventional 4x: that ratio is the knob we use to hit the budget while protecting
    depth and head dimension, both of which matter more.
    """

    # ── Vocabulary ──────────────────────────────────────────────────────────────
    # Set per language from the Phase 1 tokenizer. Both happen to be 16,000, but the
    # vocabularies are different and must never be interchanged.
    vocab_size: int = 16_000

    # ── Architecture ────────────────────────────────────────────────────────────
    d_model: int = 512          # residual stream width
    n_layers: int = 6           # number of transformer blocks
    n_heads: int = 8            # attention heads; d_model must divide evenly by this
    d_ff: int = 1536            # feed-forward inner width (3x d_model)
    max_seq_len: int = 512      # context length; hard ceiling set by the position table

    # Bonus ablation only. False removes the positional embedding table entirely, leaving the
    # causal mask as the model's only source of order information. Defaults to True so that
    # every existing config and checkpoint — none of which carries this key — loads and
    # behaves exactly as before.
    use_positional_embeddings: bool = True

    # ── Regularisation ──────────────────────────────────────────────────────────
    dropout: float = 0.1        # applied to embeddings, attention probs, sublayer outputs

    # ── Output head ─────────────────────────────────────────────────────────────
    # Tying the LM head to the token embedding saves vocab_size * d_model = 8.19M
    # parameters, a quarter of the untied model. At a 25M budget that is the difference
    # between a 6-layer stack and a 3-layer one, so it is not an optional nicety here.
    tie_embeddings: bool = True

    # ── Bookkeeping ─────────────────────────────────────────────────────────────
    language: str = ""          # "hindi" | "sindhi" — guards against loading the wrong pair
    tokenizer: str = ""         # tokenizer model filename this config was built for

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}); "
                f"each head takes a d_model/n_heads slice of the residual stream."
            )
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def d_head(self) -> int:
        """Per-head subspace width, d_model / n_heads."""
        return self.d_model // self.n_heads

    def n_params(self, non_embedding: bool = False) -> int:
        """
        Analytic parameter count, independent of any instantiated model.

        Used to sanity-check the real module tree: if the two disagree, something in the
        architecture is not what this config claims it is.
        """
        tok = self.vocab_size * self.d_model
        pos = self.max_seq_len * self.d_model if self.use_positional_embeddings else 0
        attn = 4 * (self.d_model * self.d_model + self.d_model)
        mlp = (self.d_model * self.d_ff + self.d_ff) + (self.d_ff * self.d_model + self.d_model)
        norms = 2 * (2 * self.d_model)
        blocks = self.n_layers * (attn + mlp + norms)
        total = tok + pos + blocks + 2 * self.d_model
        if not self.tie_embeddings:
            # The LM head is nn.Linear(d_model, vocab_size, bias=False), so an untied head
            # costs the weight matrix only — no per-vocabulary-entry bias term.
            total += self.d_model * self.vocab_size
        return blocks if non_embedding else total

    def to_json(self, path: Path) -> None:
        """Persist the config beside a checkpoint so runs are reproducible."""
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "GPTConfig":
        """Load a config written by to_json()."""
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
