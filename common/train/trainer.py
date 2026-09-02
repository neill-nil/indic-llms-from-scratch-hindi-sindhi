"""
Pretraining loop with checkpoint/resume for the decoder-only language models.

Written to run unchanged on the local RTX 3050 (bf16) and on Kaggle's T4/P100 (fp16 with
gradient scaling), because those GPUs predate bfloat16 support. The precision is chosen
from the device's capabilities rather than hardcoded.

Checkpointing is mandatory for this project: a run that cannot resume after an
interruption loses marks, and a ~7 hour run on a laptop or a 12-hour-capped Kaggle session
will be interrupted. Every checkpoint carries model weights, optimizer state, scheduler
position, gradient-scaler state, step counter, tokens consumed, RNG states and the config,
which is everything needed to continue as if nothing happened.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from common.data.loader import TokenDataset
from common.model.config import GPTConfig
from common.model.gpt import GPT

logger = logging.getLogger("trainer")


@dataclass
class TrainConfig:
    """
    Optimisation hyperparameters, logged alongside every checkpoint.

    Defaults follow standard practice for GPT-style models at this scale rather than a
    tuned search: with one training run per model there is no budget for a sweep, so
    conservative well-established values are safer than aggressive guesses.
    """

    # ── Token budget ────────────────────────────────────────────────────────────
    # Both models train on the same number of tokens so the H-vs-L comparison is not
    # confounded by one having seen more data. Hindi has 504.8M available and Sindhi
    # 618.4M, so 500M is the largest budget both can meet without repetition.
    max_tokens: int = 500_000_000

    # ── Batch ───────────────────────────────────────────────────────────────────
    # micro_batch is what fits in VRAM; grad_accum multiplies it up to the effective
    # batch that the optimizer actually sees. Effective batch is what matters for
    # optimisation, so the two devices use different micro_batch but the same product.
    micro_batch: int = 8
    grad_accum: int = 6          # effective batch = 8 x 6 x 512 = 24,576 tokens

    # ── Optimizer ───────────────────────────────────────────────────────────────
    learning_rate: float = 6e-4
    min_lr: float = 6e-5         # cosine floor, 10% of peak
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95          # 0.95 rather than 0.999: shorter memory suits short runs
    grad_clip: float = 1.0

    # ── Schedule ────────────────────────────────────────────────────────────────
    eval_every: int = 500
    eval_batches: int = 100      # fixed subset, fixed seed, so eval loss is comparable
    checkpoint_every: int = 1000
    log_every: int = 20

    # ── Reproducibility ─────────────────────────────────────────────────────────
    seed: int = 1337

    def tokens_per_step(self, block_size: int) -> int:
        return self.micro_batch * self.grad_accum * block_size

    def total_steps(self, block_size: int) -> int:
        return math.ceil(self.max_tokens / self.tokens_per_step(block_size))


def select_precision(device: str) -> tuple[torch.dtype, bool]:
    """
    Choose the autocast dtype for this GPU and say whether a gradient scaler is needed.

    bfloat16 has the same exponent range as float32, so gradients cannot underflow and no
    loss scaling is required. It needs Ampere (compute capability 8.0+); the local RTX
    3050 qualifies. Kaggle's T4 (7.5) and P100 (6.0) do not, so they fall back to float16,
    whose narrower exponent range does let small gradients flush to zero — hence
    GradScaler, which multiplies the loss up before backward and unscales before the step.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return torch.float32, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


class Trainer:
    """Owns the training loop, evaluation, checkpointing and resume."""

    def __init__(
        self,
        model: GPT,
        train_data: TokenDataset,
        val_data: TokenDataset,
        cfg: TrainConfig,
        out_dir: Path,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.block_size = model.config.max_seq_len
        self.total_steps = cfg.total_steps(self.block_size)

        self.amp_dtype, self.needs_scaler = select_precision(device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.needs_scaler)

        self.optimizer = self._build_optimizer()

        # Progress counters, restored on resume.
        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        # A dedicated generator for batch sampling, so its state can be checkpointed and
        # a resumed run does not replay the same windows it already trained on.
        self.data_gen = torch.Generator().manual_seed(cfg.seed)

        self.log_path = self.out_dir / "training_log.jsonl"

    # ── Optimizer ───────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        AdamW with weight decay applied only to matrices.

        Decaying biases and LayerNorm gains is counterproductive: they have one parameter
        per channel rather than per connection, so shrinking them biases the activation
        statistics rather than acting as a capacity penalty. The usual convention is to
        decay tensors with 2 or more dimensions and leave the rest alone.
        """
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        logger.info("optimizer: %d decayed tensors (%s params), %d undecayed (%s params)",
                    len(decay), f"{sum(p.numel() for p in decay):,}",
                    len(no_decay), f"{sum(p.numel() for p in no_decay):,}")

        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
            fused=self.device.startswith("cuda"),
        )

    def lr_at(self, step: int) -> float:
        """
        Linear warmup then cosine decay to min_lr.

        Warmup exists because Adam's second-moment estimate is unreliable in the first few
        hundred steps: with almost no history, a large step can be taken in a direction
        that is mostly noise, and at full learning rate that can destabilise the run before
        it starts. Cosine decay then anneals smoothly so late training makes small refining
        updates rather than continuing to bounce.
        """
        if step < self.cfg.warmup_steps:
            return self.cfg.learning_rate * (step + 1) / self.cfg.warmup_steps
        progress = (step - self.cfg.warmup_steps) / max(
            1, self.total_steps - self.cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.min_lr + coeff * (self.cfg.learning_rate - self.cfg.min_lr)

    # ── Evaluation ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Mean validation loss over a fixed set of batches.

        The generator is re-seeded every call so the same windows are scored each time.
        Otherwise the eval loss would wander with the sample rather than the model, and
        small genuine improvements would be invisible under the noise.
        """
        self.model.eval()
        gen = torch.Generator().manual_seed(self.cfg.seed + 1)
        losses = []
        for _ in range(self.cfg.eval_batches):
            x, y = self.val_data.get_batch(self.cfg.micro_batch, self.device, generator=gen)
            with torch.autocast("cuda", dtype=self.amp_dtype,
                                enabled=self.device.startswith("cuda")):
                _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    # ── Checkpointing ───────────────────────────────────────────────────────────

    def save_checkpoint(self, name: str, val_loss: Optional[float] = None) -> Path:
        """
        Write a checkpoint containing everything needed to resume bit-for-bit.

        Saved to a temporary file and renamed into place, because a process killed
        mid-write would otherwise leave a truncated file where the resume logic expects a
        valid one — and that failure only surfaces when you try to recover.
        """
        path = self.out_dir / f"{name}.pt"
        tmp = path.with_suffix(".pt.tmp")
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.cfg),
            "data_gen_state": self.data_gen.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (torch.cuda.get_rng_state()
                               if self.device.startswith("cuda") else None),
        }
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Restore a run from a checkpoint written by save_checkpoint."""
        # map_location="cpu": RNG states are ByteTensors that torch.Generator.set_state
        # and torch.set_rng_state only accept on CPU. Loading straight to CUDA silently
        # moves them to the device and both calls then raise. Weights are moved to the
        # device by load_state_dict below, so nothing is lost by staging on CPU.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        saved_cfg = GPTConfig(**ckpt["model_config"])
        if saved_cfg != self.model.config:
            raise ValueError(
                f"checkpoint architecture does not match the model being resumed:\n"
                f"  checkpoint: {saved_cfg}\n  current:    {self.model.config}")

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.data_gen.set_state(ckpt["data_gen_state"].cpu())
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and self.device.startswith("cuda"):
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        self.model.to(self.device)

        logger.info("resumed from %s at step %d (%s tokens seen)",
                    path.name, self.step, f"{self.tokens_seen:,}")

    def _log(self, record: dict) -> None:
        """Append one JSON line. JSONL survives a kill mid-write better than a JSON array."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ── Training ────────────────────────────────────────────────────────────────

    def train(self, until_step: Optional[int] = None) -> None:
        """
        Run until the token budget is exhausted.

        Args:
            until_step: stop after this step instead of at the end. The learning-rate
                schedule still spans the full run, so stopping early and resuming is
                exactly equivalent to never stopping — which is what makes the resume
                test a valid simulation of an interruption. Deriving the schedule
                horizon from the stopping point instead would silently compress the
                cosine curve and change every subsequent update.
        """
        stop_at = self.total_steps if until_step is None else min(until_step, self.total_steps)
        self.model.train()
        cfg = self.cfg
        tokens_per_step = cfg.tokens_per_step(self.block_size)

        logger.info("%s", "=" * 68)
        logger.info("training %s | %s params | ctx %d",
                    self.model.config.language,
                    f"{self.model.num_parameters():,}", self.block_size)
        logger.info("  precision      : %s%s", self.amp_dtype,
                    " + GradScaler" if self.needs_scaler else "")
        logger.info("  micro batch    : %d x %d accum = %s tokens/step",
                    cfg.micro_batch, cfg.grad_accum, f"{tokens_per_step:,}")
        logger.info("  token budget   : %s over %s steps",
                    f"{cfg.max_tokens:,}", f"{self.total_steps:,}")
        logger.info("  starting at    : step %d", self.step)
        logger.info("%s", "=" * 68)

        t_start = time.time()
        t_window = time.time()

        while self.step < stop_at:
            lr = self.lr_at(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0

            for micro in range(cfg.grad_accum):
                x, y = self.train_data.get_batch(
                    cfg.micro_batch, self.device, generator=self.data_gen)
                with torch.autocast("cuda", dtype=self.amp_dtype,
                                    enabled=self.device.startswith("cuda")):
                    _, loss = self.model(x, y)
                    # Divide so the accumulated gradient is the mean over the effective
                    # batch, not the sum — otherwise the effective learning rate would
                    # scale with grad_accum.
                    loss = loss / cfg.grad_accum
                self.scaler.scale(loss).backward()
                accumulated += loss.item()

            # Unscale before clipping: with fp16 the gradients are still multiplied by the
            # scaler's factor, so clipping to 1.0 without unscaling would clip the wrong
            # quantity by a large and varying factor.
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1
            self.tokens_seen += tokens_per_step

            if self.step % cfg.log_every == 0:
                dt = time.time() - t_window
                tok_s = cfg.log_every * tokens_per_step / dt
                elapsed = time.time() - t_start
                remaining = (self.total_steps - self.step) * tokens_per_step / max(tok_s, 1)
                logger.info(
                    "step %6d/%d | loss %.4f | lr %.2e | grad %.2f | %s tok/s | "
                    "%.1f%% | eta %.1fh",
                    self.step, self.total_steps, accumulated, lr, grad_norm,
                    f"{int(tok_s):,}", 100 * self.tokens_seen / cfg.max_tokens,
                    remaining / 3600)
                self._log({"step": self.step, "tokens": self.tokens_seen,
                           "train_loss": accumulated, "lr": lr,
                           "grad_norm": float(grad_norm), "tokens_per_sec": tok_s,
                           "elapsed_s": elapsed})
                t_window = time.time()

            if self.step % cfg.eval_every == 0 or self.step == stop_at:
                val_loss = self.evaluate()
                ppl = math.exp(min(val_loss, 20))
                logger.info("  eval @ %d: val loss %.4f | perplexity %.2f",
                            self.step, val_loss, ppl)
                self._log({"step": self.step, "tokens": self.tokens_seen,
                           "val_loss": val_loss, "val_perplexity": ppl})
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best", val_loss)
                    logger.info("  new best (%.4f) -> best.pt", val_loss)
                t_window = time.time()

            if self.step % cfg.checkpoint_every == 0:
                self.save_checkpoint("latest")
                t_window = time.time()

        self.save_checkpoint("latest")
        hours = (time.time() - t_start) / 3600
        if self.step >= self.total_steps:
            logger.info("done: %s tokens in %.2f h | best val loss %.4f (ppl %.2f)",
                        f"{self.tokens_seen:,}", hours, self.best_val_loss,
                        math.exp(min(self.best_val_loss, 20)))
        else:
            logger.info("stopped early at step %d/%d after %.2f h; resume to continue",
                        self.step, self.total_steps, hours)
