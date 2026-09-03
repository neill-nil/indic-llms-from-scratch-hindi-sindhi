"""
Kaggle training cell. Set LANGUAGE below to pick the model.

Paste this into a single Kaggle notebook cell with a GPU attached. No internet access is
required: the repository is private, so the code arrives as an uploaded Dataset rather
than a git clone.

The cell is idempotent. Re-running it after a session times out resumes from the last
checkpoint instead of starting over, which matters because Kaggle sessions are capped at
roughly 12 hours and can drop without warning.

Precision is chosen from the device. Kaggle's T4 (compute capability 7.5) and P100 (6.0)
predate bfloat16, so common.train.trainer falls back to float16 with a GradScaler; the
identical code runs in bfloat16 on the local Ampere card.

SETUP, once:
  1. Upload two Datasets (built by kaggle/package_*.sh):
       <language>-tokens   the .bin files plus that language's tokenizer
       lma-code    ~56 KB  the training code, carrying both model configs
  2. Settings -> Accelerator -> GPU T4 x2 (or P100)
  3. Add BOTH datasets in the right-hand "Input" panel.
  Internet can stay off.

TO RESUME after a session ends:
  1. Add this notebook's previous version output as an input dataset.
  2. Set RESUME_FROM below to the checkpoint inside it.
  3. Re-run the cell.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LANGUAGE = "hindi"          # "hindi" or "sindhi"

# ── Paths ───────────────────────────────────────────────────────────────────────
# Both datasets are located by the files they contain rather than by a fixed path, because
# Kaggle mounts them at different places depending on how they were attached
# (/kaggle/input/<slug> or /kaggle/input/datasets/<user>/<slug>). Set an override only if
# auto-discovery picks the wrong directory.
DATA_DIR_OVERRIDE = None
CODE_DIR_OVERRIDE = None
OUT_DIR = None              # defaults to /kaggle/working/<language>_run

# Set to a checkpoint path to continue a previous session; None starts fresh.
RESUME_FROM = None

# ── Batch. Must match the Hindi run's effective batch. ──────────────────────────
# The H-vs-L comparison depends on both models seeing the same optimisation, not merely
# the same quantity of data. Sindhi's context is 1024 against Hindi's 512, so the sequence
# count is halved to compensate:
#
#     Hindi   micro 8 x accum 6 x ctx  512 = 24,576 tokens/step
#     Sindhi  micro 8 x accum 3 x ctx 1024 = 24,576 tokens/step
#
# The T4's 16 GB would allow a larger micro-batch; grad_accum absorbs that, not the
# effective batch.
# grad_accum is DERIVED from the context length rather than set by hand, so both models
# take the same optimizer step no matter which one this cell is training:
#
#     Hindi   micro 8 x accum 6 x ctx  512 = 24,576 tokens/step
#     Sindhi  micro 8 x accum 3 x ctx 1024 = 24,576 tokens/step
#
# Setting it manually is how a mismatched effective batch sneaks in.
MICRO_BATCH = 8
MAX_TOKENS = 500_000_000
EXPECTED_TOKENS_PER_STEP = 24_576


def find_dataset(markers, label, override=None):
    """
    Locate an attached Kaggle dataset by the files it contains.

    Kaggle's mount path varies — it can be /kaggle/input/<slug>, or nested under
    /kaggle/input/datasets/<user>/<slug>, depending on how the dataset was attached.
    Searching for known marker files is more reliable than hardcoding any one convention,
    and it keeps working if the dataset is renamed.
    """
    if override:
        p = Path(override)
        if all((p / m).exists() for m in markers):
            return p
        raise SystemExit(f"{label}: override {override} does not contain {markers}")

    root = Path("/kaggle/input")
    if not root.exists():
        raise SystemExit(f"{label}: /kaggle/input does not exist — attach your datasets.")

    hits = []
    for candidate in [root, *(d for d in root.rglob("*") if d.is_dir())]:
        if all((candidate / m).exists() for m in markers):
            hits.append(candidate)
    if not hits:
        listing = "\n".join(f"    {p}" for p in sorted(root.rglob("*"))[:40]) or "    (empty)"
        raise SystemExit(
            f"{label}: could not find a directory containing {markers}.\n"
            f"  Attached under /kaggle/input:\n{listing}")
    # Shallowest match wins, so a nested copy never shadows the real one.
    best = min(hits, key=lambda p: len(p.parts))
    print(f"{label}: found at {best}")
    return best


# ── Stage the code into a writable scratch directory ────────────────────────────
# /kaggle/input is read-only, so the code is copied out rather than imported in place.
# It goes to /tmp rather than /kaggle/working because Kaggle captures all of
# /kaggle/working as the notebook's saved output: staging there would put the 1.2 GB
# symlinks into /kaggle/input inside the captured tree and bury the checkpoints among
# scratch files. /tmp is writable and not captured.
REPO_DIR = Path("/tmp/lma_repo")
code_src = find_dataset(["common/model/gpt.py"], "code", CODE_DIR_OVERRIDE)

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
shutil.copytree(code_src, REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print(f"code staged: {code_src} -> {REPO_DIR}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece>=0.2.0"],
               check=True)

# ── Link the uploaded token data into the layout the trainer expects ────────────
data_src = find_dataset(["train.bin", f"{LANGUAGE}_unigram_16000.model"],
                        "token data", DATA_DIR_OVERRIDE)

tokenized = REPO_DIR / LANGUAGE / "data" / "tokenized"
tokenized.mkdir(parents=True, exist_ok=True)
for name in ("train.bin", "val.bin", "test.bin"):
    src = data_src / name
    if not src.exists():
        raise SystemExit(
            f"missing {src}. Did the archive extract flat?\n"
            f"Found in {data_src}: {sorted(p.name for p in data_src.iterdir())}")
    target = tokenized / name
    if not target.exists():
        # COPY, do not symlink. /kaggle/input is a network-backed filesystem, and the
        # loader memory-maps this file and reads a random window per sequence — 24 random
        # seeks per optimizer step. Served over the network that is I/O-bound and starves
        # the GPU; measured at ~10k tok/s against ~25k on a far weaker local card. Copying
        # to local disk first costs about a minute and removes the bottleneck.
        print(f"  copying {name} to local disk ({src.stat().st_size/1e9:.2f} GB)...",
              flush=True)
        shutil.copy(src, target)

tok_dir = REPO_DIR / LANGUAGE / "tokenizer"
tok_dir.mkdir(parents=True, exist_ok=True)
for name in (f"{LANGUAGE}_unigram_16000.model", f"{LANGUAGE}_unigram_16000.vocab"):
    src = data_src / name
    if not src.exists():
        raise SystemExit(f"missing {src} — the tokenizer must be in the dataset.")
    shutil.copy(src, tok_dir / name)

print("data linked:")
for f in sorted(tokenized.iterdir()):
    print(f"  {f.name:12s} {os.path.getsize(f)/1e6:9.1f} MB")

# ── Stage a checkpoint to resume from ───────────────────────────────────────────
out = Path(OUT_DIR or f"/kaggle/working/{LANGUAGE}_run")
out.mkdir(parents=True, exist_ok=True)
resume = False
if RESUME_FROM:
    src = Path(RESUME_FROM)
    if not src.exists():
        raise SystemExit(f"RESUME_FROM does not exist: {src}")
    shutil.copy(src, out / "latest.pt")   # /kaggle/input is read-only
    resume = True
    print(f"resuming from {src}")
elif (out / "latest.pt").exists():
    resume = True
    print("found a checkpoint in the working directory; resuming")

# ── Guard the effective batch ───────────────────────────────────────────────────
# Read the context length from the committed config rather than assuming it, and refuse
# to start if the batch settings would not match the Hindi run.
_ctx = json.loads(
    (REPO_DIR / LANGUAGE / "configs" / "model_config.json").read_text())["max_seq_len"]
if EXPECTED_TOKENS_PER_STEP % (MICRO_BATCH * _ctx):
    raise SystemExit(
        f"{EXPECTED_TOKENS_PER_STEP:,} tokens/step is not divisible by "
        f"micro_batch {MICRO_BATCH} x ctx {_ctx}; adjust MICRO_BATCH.")
GRAD_ACCUM = EXPECTED_TOKENS_PER_STEP // (MICRO_BATCH * _ctx)
print(f"effective batch: {MICRO_BATCH} x {GRAD_ACCUM} x {_ctx} = "
      f"{MICRO_BATCH * GRAD_ACCUM * _ctx:,} tokens/step "
      f"(identical for both models by construction)")

# ── Report the environment ──────────────────────────────────────────────────────
import torch
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then re-run.")
print(f"device: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()} "
      f"(False on T4/P100 -> fp16 + GradScaler)")

# ── Train ───────────────────────────────────────────────────────────────────────
cmd = [sys.executable, "-m", "common.train.train",
       "--language", LANGUAGE,
       "--out_dir", str(out),
       "--max_tokens", str(MAX_TOKENS),
       "--micro_batch", str(MICRO_BATCH),
       "--grad_accum", str(GRAD_ACCUM),
       "--checkpoint_every", "500"]      # frequent: sessions drop without warning
if resume:
    cmd.append("--resume")

print("\nrunning:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=str(REPO_DIR), check=True)

# ── Artefacts ───────────────────────────────────────────────────────────────────
# /kaggle/working is saved as the notebook's output when the session ends. Download
# best.pt, latest.pt, training_log.jsonl and train.log — the logs are required in the
# repository for the report's loss curves, and are easy to forget once you have weights.
print(f"\nartefacts in {out}:")
for f in sorted(out.iterdir()):
    print(f"  {f.name:24s} {f.stat().st_size/1e6:9.1f} MB")
