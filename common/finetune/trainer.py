"""
Reasoning finetuning: a thin specialisation of the Phase 2 pretraining trainer.

Everything that made pretraining recoverable — checkpoint contents, atomic writes, RNG
snapshots at step boundaries, signal handling, precision selection — is inherited rather
than reimplemented, so a finetuned checkpoint resumes by exactly the same mechanism as a
pretrained one. The brief requires finetuned checkpoints in "the same resume-capable
format as pretraining"; sharing the code is the only way to be sure of that rather than
hoping two implementations agree.

Only two things genuinely differ.

**Initialisation.** Finetuning starts from a pretrained checkpoint's *weights alone*. The
optimizer state is deliberately discarded: Adam's moment estimates were accumulated under
a different objective at a learning rate an order of magnitude higher, and carrying them
over would apply that stale momentum to the first finetuning steps.

**Evaluation.** Validation loss on masked answer tokens is a poor thing to watch on its
own, because the quantity we actually care about is whether the whole answer comes out
right. `evaluate()` therefore also reports the fraction of examples whose every answer
token is predicted correctly under teacher forcing.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from common.finetune.dataset import IGNORE_INDEX, ReasoningDataset
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.trainer import TrainConfig, Trainer

logger = logging.getLogger("finetune.trainer")


@dataclass
class FinetuneConfig(TrainConfig):
    """
    Optimisation settings for finetuning.

    The learning rate is 1e-4 against pretraining's 6e-4. Finetuning starts from weights
    that already encode the language; a peak rate as high as pretraining's would take
    large steps away from that solution in the first few hundred updates, on a narrow
    task, and the usual result is that general language ability is destroyed while the
    task is learned. Warmup is short for the same reason — there is no cold-start
    instability to protect against when the weights are already good.

    `max_tokens` is inherited but not meaningful here; the run length is set in epochs
    over a finite dataset, and `total_steps` is overridden accordingly.
    """

    learning_rate: float = 1e-4
    min_lr: float = 1e-5           # 10% of peak; run.py re-derives it from the chosen rate
    warmup_steps: int = 100
    weight_decay: float = 0.01     # lighter than pretraining's 0.1: a small dataset and a
                                   # short run give the model little chance to overfit by
                                   # weight growth, and heavy decay would mostly just pull
                                   # the pretrained solution apart
    micro_batch: int = 32
    grad_accum: int = 1
    epochs: int = 3

    eval_every: int = 200
    eval_batches: int = 40
    checkpoint_every: int = 500
    log_every: int = 25


class FinetuneTrainer(Trainer):
    """Trainer whose data is prompt/answer pairs with a masked loss."""

    def __init__(self, model: GPT, train_data: ReasoningDataset,
                 val_data: ReasoningDataset, cfg: FinetuneConfig,
                 out_dir: Path, device: str = "cuda"):
        super().__init__(model, train_data, val_data, cfg, out_dir, device)

        # The parent derives a step count from a token budget. Here the run is defined by
        # passes over a finite dataset, so the schedule horizon is recomputed — the cosine
        # curve must span the real run or the learning rate never reaches its floor.
        examples_per_step = cfg.micro_batch * cfg.grad_accum
        self.steps_per_epoch = max(1, len(train_data) // examples_per_step)
        self.total_steps = self.steps_per_epoch * cfg.epochs
        self.block_size = train_data.max_len

        # The parent reports progress as a fraction of `max_tokens`, which it inherits as
        # the 500M pretraining budget. Left alone, a finetuning run would report 0.3%
        # complete at the moment it finished. Restated as what this run will actually
        # consume.
        self.cfg.max_tokens = self.total_steps * cfg.tokens_per_step(self.block_size)

    def _log(self, record: dict) -> None:
        """
        Attach accuracy to every evaluation record.

        The parent's training loop writes the evaluation record itself, so this is the
        hook that gets accuracy into the log and onto the console without duplicating the
        loop. Checkpoint selection is left on validation loss: with roughly two answer
        tokens per example the two move together closely, and loss is the finer-grained
        signal of the pair.
        """
        if "val_loss" in record:
            record["val_accuracy"] = self.last_accuracy
            logger.info("  eval @ %d: exact-answer accuracy %.4f (teacher forced)",
                        record.get("step", self.step), self.last_accuracy)
        super()._log(record)

    def init_from_pretrained(self, path: Path) -> None:
        """
        Load model weights from a pretrained checkpoint, and nothing else.

        Step counter, optimizer moments, scheduler position and RNG state all stay at
        their fresh values, so this is the start of a new run that happens to begin from
        trained weights — not a continuation of the pretraining run.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved_cfg = GPTConfig(**ckpt["model_config"])
        if saved_cfg != self.model.config:
            raise ValueError(
                f"pretrained checkpoint architecture does not match:\n"
                f"  checkpoint: {saved_cfg}\n  current:    {self.model.config}")

        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        logger.info("initialised from %s (pretrained step %d, val loss %.4f) — "
                    "weights only, fresh optimizer",
                    path.name, ckpt.get("step", -1),
                    ckpt.get("best_val_loss", float("nan")))

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Mean loss over answer tokens, plus exact-answer accuracy under teacher forcing.

        Accuracy is the number actually worth watching: loss can improve while the
        argmax stays wrong, and the reported metric in Phase 3 is exact match. It is
        computed teacher-forced (each position scored against the gold prefix) rather
        than by generating, because generating on every evaluation would dominate the
        run time. It is a close upper-bound proxy for greedy exact match; the real greedy
        numbers come from `common.finetune.evaluate` at the end.

        The whole validation split is scored, not a sample, so the curve moves with the
        model rather than with the draw.
        """
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        n_correct, n_examples = 0, 0

        for x, y in self.val_data.iter_sequential(self.cfg.micro_batch, self.device):
            with torch.autocast("cuda", dtype=self.amp_dtype,
                                enabled=self.device.startswith("cuda")):
                logits, loss = self.model(x, y)
            total_loss += loss.item()
            n_batches += 1

            supervised = y != IGNORE_INDEX
            correct = (logits.argmax(-1) == y) & supervised
            # An example counts only if EVERY supervised token is right — a partially
            # correct answer is a wrong answer.
            per_example = correct.sum(1) == supervised.sum(1)
            n_correct += int(per_example.sum())
            n_examples += x.size(0)

        self.model.train()
        self.last_accuracy = n_correct / max(1, n_examples)
        return total_loss / max(1, n_batches)

    def train(self, until_step: Optional[int] = None) -> None:
        """Run finetuning, logging accuracy alongside loss."""
        self.last_accuracy = 0.0
        logger.info("finetuning %s | %d examples | %d steps/epoch x %d epochs = %d steps",
                    self.model.config.language, len(self.train_data),
                    self.steps_per_epoch, self.cfg.epochs, self.total_steps)
        super().train(until_step)
