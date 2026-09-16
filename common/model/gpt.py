"""
The complete decoder-only Transformer language model.

Stage 4: assembles the pieces built in stages 1-3 into a model that maps token ids to
next-token logits, and computes the causal language-modelling loss.

    ids -> InputEmbedding -> N x TransformerBlock -> final LayerNorm -> LM head -> logits
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model.config import GPTConfig
from common.model.embeddings import InputEmbedding
from common.model.block import TransformerBlock


class GPT(nn.Module):
    """
    Decoder-only Transformer language model.

    Shapes
    ------
        idx     (B, T)              int64 token ids
        logits  (B, T, vocab_size)  next-token scores; logits[b, t] predicts idx[b, t+1]

    Weight tying
    ------------
    The output projection reuses the token embedding matrix rather than learning its own.
    The two are genuinely the same tensor, not a copy, so a gradient from either end
    updates the same weights.

    The saving is vocab_size x d_model = 16,000 x 512 = 8,192,000 parameters, which is
    a quarter of the untied model and a third of our ~25M budget. Spent on the transformer
    stack instead, that is the difference between a 6-layer model and roughly a 3-layer
    one, so at this scale tying is not a minor economy — it is what makes the depth
    affordable.

    There is a linguistic argument as well as an economic one. The input embedding answers
    "what does this token mean" and the output projection answers "which token fits this
    context"; both are maps between the vocabulary and the same representation space, so
    forcing them to share weights is a reasonable prior rather than a pure compression
    hack. It is standard practice from GPT-2 onward.

    Why a final LayerNorm
    ---------------------
    Because the blocks are pre-norm, nothing ever normalises the residual stream itself —
    each block only adds to it. Without a final normalisation the stream's scale entering
    the output projection would depend on depth and on how far training has progressed,
    and the tied head would see inputs whose magnitude drifts over the run. `ln_f` pins
    that scale down.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embeddings = InputEmbedding(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.d_model)

        # No bias: the tied embedding has none, and a per-vocabulary-entry bias would add
        # another vocab_size parameters for little benefit.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Assignment, not copy — `is` identity is asserted in the tests. This must
            # happen after both modules are constructed, or the head's own initialisation
            # would be the one that survives.
            self.lm_head.weight = self.embeddings.token_embedding.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ── Parameter accounting ────────────────────────────────────────────────────

    def num_parameters(self, non_embedding: bool = False) -> int:
        """
        Count trainable parameters.

        nn.Module.parameters() de-duplicates shared tensors, so a tied head is counted
        once — which is the honest number, since there is only one matrix.

        Args:
            non_embedding: exclude the token and position tables, leaving the transformer
                stack and final norm. This is the figure to compare across models whose
                vocabularies or context lengths differ.
        """
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embeddings.token_embedding.weight.numel()
            if self.embeddings.position_embedding is not None:
                total -= self.embeddings.position_embedding.weight.numel()
        return total

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Args:
            idx: (B, T) token ids.
            targets: (B, T) next-token ids, already shifted by the data loader so that
                targets[b, t] is the token following idx[b, t]. Shifting in the loader
                rather than here means no position is wasted: with a (T+1)-token slice the
                loader produces T full training pairs, whereas shifting internally would
                discard the last position of every window.
            return_attention: also return per-layer attention weights, a tuple of
                n_layers tensors of shape (B, n_heads, T, T). Used by the attention
                analysis; leave off during training, as it holds every layer's T x T
                matrix in memory.

        Returns:
            (logits, loss) where loss is None if targets is None, optionally followed by
            the attention tuple.
        """
        x = self.embeddings(idx)

        attentions = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy between the logits at position t and the token at t+1,
            # averaged over every position in the batch. Flattened to (B*T, V) and (B*T,)
            # because F.cross_entropy expects a single class axis.
            # ignore_index=-100 lets a caller exclude positions from the loss. Pretraining
            # never uses it: every position of a packed token stream is a genuine training
            # target, so no target is ever -100 and the behaviour is unchanged. Reasoning
            # finetuning needs it, because an example is a prompt followed by an answer and
            # only the answer should be learned — training the model to reproduce the
            # question would spend most of its capacity on text it will always be given.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        if return_attention:
            return logits, loss, tuple(attentions)
        return logits, loss

    # ── Generation ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively extend a prompt.

        Args:
            idx: (B, T) prompt token ids.
            max_new_tokens: how many tokens to append.
            temperature: divides the logits before softmax. Below 1 sharpens the
                distribution towards the argmax; above 1 flattens it. At exactly 0 we take
                the argmax directly, since dividing by zero is undefined — that is greedy
                decoding.
            top_k: if set, sample only from the k highest-scoring tokens.
            eos_id: stop early once every sequence in the batch has emitted this token.

        Returns:
            (B, T + n) ids, where n <= max_new_tokens.
        """
        was_training = self.training
        self.eval()

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # The position table only has max_seq_len rows, so a prompt longer than the
            # context must be cropped to its most recent window.
            window = idx[:, -self.config.max_seq_len:]
            logits, _ = self(window)
            logits = logits[:, -1, :]                # (B, vocab) — only the last step

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            if eos_id is not None:
                # Once a sequence is finished keep emitting eos so the tensor stays
                # rectangular; callers truncate at the first eos.
                next_id = torch.where(finished.unsqueeze(1),
                                      torch.full_like(next_id, eos_id), next_id)
                finished |= next_id.squeeze(1) == eos_id

            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool(finished.all()):
                break

        if was_training:
            self.train()
        return idx

    # ── Construction helpers ────────────────────────────────────────────────────

    @classmethod
    def from_config_file(cls, path: Path) -> "GPT":
        """Build a model from a saved config, e.g. hindi/configs/model_config.json."""
        return cls(GPTConfig.from_json(path))

    def parameter_summary(self) -> str:
        """Human-readable breakdown, logged at the start of a training run."""
        c = self.config
        tok = c.vocab_size * c.d_model
        pos = c.max_seq_len * c.d_model if c.use_positional_embeddings else 0
        blocks = sum(p.numel() for p in self.blocks.parameters())
        total = self.num_parameters()
        lines = [
            f"{'component':<26}{'params':>13}{'share':>9}",
            f"{'-' * 48}",
            f"{'token embedding':<26}{tok:>13,}{tok / total:>9.1%}",
            f"{'position embedding':<26}{pos:>13,}{pos / total:>9.1%}",
            f"{f'{c.n_layers} transformer blocks':<26}{blocks:>13,}{blocks / total:>9.1%}",
            f"{'final LayerNorm':<26}{2 * c.d_model:>13,}{2 * c.d_model / total:>9.1%}",
            f"{'LM head':<26}{'0 (tied)' if c.tie_embeddings else f'{tok:,}':>13}"
            f"{0 if c.tie_embeddings else tok / total:>9.1%}",
            f"{'-' * 48}",
            f"{'total':<26}{total:>13,}",
            f"{'non-embedding':<26}{self.num_parameters(non_embedding=True):>13,}",
        ]
        return "\n".join(lines)
