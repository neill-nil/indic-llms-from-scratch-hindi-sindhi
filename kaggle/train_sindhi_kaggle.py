"""
Kaggle training cell for Model L (Sindhi).

Paste this into a single Kaggle notebook cell, with GPU and internet enabled. It clones
the repository, locates the uploaded token data, and starts or resumes training.

The script is deliberately idempotent: running the same cell again after a session times
out resumes from the last checkpoint rather than starting over. That matters because
Kaggle sessions are capped at roughly 12 hours and can drop earlier, and this run is long
enough to be interrupted at least once.

Precision is chosen automatically. Kaggle's T4 (compute capability 7.5) and P100 (6.0)
predate bfloat16, so common.train.trainer falls back to float16 with a GradScaler; the
same code runs in bfloat16 on the local Ampere card without modification.

SETUP, once:
  1. Add your token data as a Kaggle Dataset (see kaggle/PACKAGE.md for how to build it).
  2. Settings -> Accelerator -> GPU T4 x2 (or P100)
  3. Settings -> Internet -> On
  4. Add the dataset in the right-hand "Input" panel.

TO RESUME after a session ends:
  1. Add this notebook's previous version output as an input dataset.
  2. Set RESUME_FROM below to that path, e.g.
       /kaggle/input/sindhi-run-v1/latest.pt
  3. Run the cell again.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Settings you may need to change ─────────────────────────────────────────────
REPO_URL = "https://github.com/CL3-410/individual-project-neill-nil.git"
REPO_BRANCH = "phase-2"

# Where the uploaded .bin files live. Kaggle mounts datasets under /kaggle/input/<slug>/.
DATA_DIR = "/kaggle/input/sindhi-tokens"

# Set to a checkpoint path to continue a previous session; leave None to start fresh.
RESUME_FROM = None

OUT_DIR = "/kaggle/working/sindhi_run"

# The effective batch must match the Hindi run at 24,576 tokens per optimizer step: the
# H-vs-L comparison depends on the two models seeing the same optimisation, not merely the
# same quantity of data. Sindhi's context is 1024 rather than 512, so the sequence count
# is halved to compensate:
#
#     Hindi   micro 8 x accum 6 x ctx  512 = 24,576 tokens/step
#     Sindhi  micro 8 x accum 3 x ctx 1024 = 24,576 tokens/step
#
# The T4's 16 GB would allow a larger micro-batch; grad_accum is what absorbs that, not
# the effective batch.
MICRO_BATCH = 8
GRAD_ACCUM = 3
MAX_TOKENS = 500_000_000
EXPECTED_TOKENS_PER_STEP = 24_576

# ── Clone the repository ────────────────────────────────────────────────────────
REPO_DIR = Path("/kaggle/working/repo")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
                    REPO_URL, str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR))

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentencepiece>=0.2.0"], check=True)

# ── Point the repo layout at the uploaded data ──────────────────────────────────
# The trainer resolves paths relative to the repo root, so the uploaded token files and
# the tokenizer are linked into the places it expects rather than passing paths around.
data_src = Path(DATA_DIR)
tokenized = REPO_DIR / "sindhi" / "data" / "tokenized"
tokenized.mkdir(parents=True, exist_ok=True)
for name in ("train.bin", "val.bin", "test.bin"):
    src = data_src / name
    if src.exists() and not (tokenized / name).exists():
        os.symlink(src, tokenized / name)

tok_dir = REPO_DIR / "sindhi" / "tokenizer"
tok_dir.mkdir(parents=True, exist_ok=True)
for name in ("sindhi_unigram_16000.model", "sindhi_unigram_16000.vocab"):
    src = data_src / name
    if src.exists() and not (tok_dir / name).exists():
        shutil.copy(src, tok_dir / name)

# ── Stage a checkpoint to resume from ───────────────────────────────────────────
out = Path(OUT_DIR)
out.mkdir(parents=True, exist_ok=True)
resume = False
if RESUME_FROM:
    src = Path(RESUME_FROM)
    if not src.exists():
        raise FileNotFoundError(f"RESUME_FROM does not exist: {src}")
    # Kaggle input is read-only, so copy into the writable working directory.
    shutil.copy(src, out / "latest.pt")
    resume = True
    print(f"staged {src} -> {out / 'latest.pt'}")
elif (out / "latest.pt").exists():
    resume = True
    print("found an existing checkpoint in the working directory; resuming")

# ── Guard the effective batch ───────────────────────────────────────────────────
# Sindhi's context length lives in its committed model config, so read it rather than
# assuming, and fail loudly if the batch settings would not match the Hindi run.
import json
_ctx = json.loads((REPO_DIR / "sindhi" / "configs" / "model_config.json").read_text())["max_seq_len"]
_tokens_per_step = MICRO_BATCH * GRAD_ACCUM * _ctx
if _tokens_per_step != EXPECTED_TOKENS_PER_STEP:
    raise SystemExit(
        f"effective batch is {_tokens_per_step:,} tokens/step but the Hindi run uses "
        f"{EXPECTED_TOKENS_PER_STEP:,}. With ctx={_ctx}, set MICRO_BATCH x GRAD_ACCUM = "
        f"{EXPECTED_TOKENS_PER_STEP // _ctx}.")
print(f"effective batch: {MICRO_BATCH} x {GRAD_ACCUM} x {_ctx} = {_tokens_per_step:,} "
      f"tokens/step (matches the Hindi run)")

# ── Report the environment ──────────────────────────────────────────────────────
import torch
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()} "
          f"(False on T4/P100 -> fp16 + GradScaler)")

# ── Train ───────────────────────────────────────────────────────────────────────
cmd = [sys.executable, "-m", "common.train.train",
       "--language", "sindhi",
       "--out_dir", OUT_DIR,
       "--max_tokens", str(MAX_TOKENS),
       "--micro_batch", str(MICRO_BATCH),
       "--grad_accum", str(GRAD_ACCUM),
       "--checkpoint_every", "500"]     # frequent: sessions can drop without warning
if resume:
    cmd.append("--resume")

print("running:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=str(REPO_DIR), check=True)

# ── Copy the artefacts somewhere they persist ───────────────────────────────────
# /kaggle/working is saved as the notebook's output when the session ends.
print("\ncheckpoints in /kaggle/working/sindhi_run:")
for f in sorted(out.iterdir()):
    print(f"  {f.name:24s} {f.stat().st_size/1e6:8.1f} MB")
