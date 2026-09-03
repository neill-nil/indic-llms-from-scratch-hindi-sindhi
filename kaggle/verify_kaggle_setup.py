"""
Kaggle pre-flight check. Run this cell BEFORE the training cell.

It takes about a minute and confirms the datasets are attached, extracted where the
trainer expects, internally consistent, and that the model builds and takes a training
step on the GPU. Finding a wrong path here costs a minute; finding it four hours into a
run costs four hours.

Paste into its own cell and run.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

LANGUAGE = "hindi"          # "hindi" or "sindhi" — must match the training cell

# Leave these as None to auto-discover. Set a path only to override.
DATA_DIR_OVERRIDE = None
CODE_DIR_OVERRIDE = None

ok = True


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




def check(label, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{'  ' + detail if detail else ''}")


# ── 1. What is actually attached ────────────────────────────────────────────────
print("=" * 70)
print("1. Datasets attached to this notebook")
print("=" * 70)
root = Path("/kaggle/input")
if not root.exists():
    print("  /kaggle/input does not exist — no datasets attached at all.")
    raise SystemExit(1)

for d in sorted(root.iterdir()):
    files = sorted(d.rglob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    print(f"\n  /kaggle/input/{d.name}/   ({total/1e6:,.1f} MB)")
    for f in files[:12]:
        if f.is_file():
            print(f"      {f.relative_to(d)}  ({f.stat().st_size/1e6:,.1f} MB)")
    if len(files) > 12:
        print(f"      ... and {len(files) - 12} more")

# ── 2. Expected paths ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. Expected paths")
print("=" * 70)
DATA_DIR = find_dataset(["train.bin", f"{LANGUAGE}_unigram_16000.model"],
                        "token data", DATA_DIR_OVERRIDE)
CODE_DIR = find_dataset(["common/model/gpt.py"], "code", CODE_DIR_OVERRIDE)
check("token dataset located", True, str(DATA_DIR))
check("code dataset located", True, str(CODE_DIR))

# Expected counts come from the tokenization_stats.json shipped inside the dataset, so
# this check works for either language without hardcoded numbers.
import json as _json
stats_path = Path(DATA_DIR) / "tokenization_stats.json"
check("tokenization_stats.json present", stats_path.exists())
EXPECTED = {}
if stats_path.exists():
    _st = _json.loads(stats_path.read_text())
    check("stats are for this language", _st.get("language") == LANGUAGE,
          f"stats say '{_st.get('language')}', cell says '{LANGUAGE}'")
    EXPECTED = {k: v["tokens"] for k, v in _st.get("splits", {}).items()}
    for split, n in EXPECTED.items():
        p = Path(DATA_DIR) / f"{split}.bin"
        expect_bytes = n * 2          # uint16
        check(f"{split}.bin size", p.exists() and p.stat().st_size == expect_bytes,
              f"{p.stat().st_size/1e6:,.1f} MB (expected {expect_bytes/1e6:,.1f})"
              if p.exists() else "missing")

# ── 3. Token data is intact ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. Token data integrity")
print("=" * 70)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece>=0.2.0"],
               check=True)
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor(
    model_file=str(Path(DATA_DIR) / f"{LANGUAGE}_unigram_16000.model"))
check("tokenizer loads", sp.get_piece_size() == 16000, f"vocab {sp.get_piece_size():,}")
sep = sp.piece_to_id("<|endoftext|>")
check("separator token present", sep != sp.unk_id(), f"id {sep}")

for split, expect in EXPECTED.items():
    a = np.memmap(Path(DATA_DIR) / f"{split}.bin", dtype=np.uint16, mode="r")
    check(f"{split}.bin token count", len(a) == expect,
          f"{len(a):,} (expected {expect:,})")
    check(f"{split}.bin ids within vocab", int(a.max()) < 16000, f"max id {int(a.max())}")

a = np.memmap(Path(DATA_DIR) / "val.bin", dtype=np.uint16, mode="r")
sample = sp.decode([int(t) for t in a[5000:5060] if int(t) != sep])
print(f"\n  decoded sample (should be readable {LANGUAGE}):\n    {sample[:100]}")
check("sample decodes to non-empty text", len(sample.strip()) > 20)

# ── 4. Code imports and the model builds ────────────────────────────────────────
print("\n" + "=" * 70)
print("4. Code and model")
print("=" * 70)
REPO_DIR = Path("/tmp/lma_repo_check")   # /tmp is not captured as output
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
shutil.copytree(CODE_DIR, REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

import torch
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.trainer import select_precision

cfg = GPTConfig.from_json(REPO_DIR / LANGUAGE / "configs" / "model_config.json")
check("config loads", cfg.language == LANGUAGE, f"ctx {cfg.max_seq_len}")
check("vocab matches tokenizer", cfg.vocab_size == sp.get_piece_size())

model = GPT(cfg)
check("model builds", 24_000_000 < model.num_parameters() < 25_000_000,
      f"{model.num_parameters():,} params")

# ── 5. GPU and a real training step ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. GPU and one training step")
print("=" * 70)
check("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"      device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    dtype, needs_scaler = select_precision("cuda")
    print(f"      precision: {dtype}{' + GradScaler' if needs_scaler else ''}")

    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    x = torch.randint(0, cfg.vocab_size, (8, cfg.max_seq_len), device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        _, loss = model(x, x)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1e9
    check("one fwd+bwd+step completes", torch.isfinite(loss),
          f"loss {loss.item():.3f}, peak VRAM {peak:.2f} GB")
    # An untrained model should sit near ln(vocab) = 9.68; far from it means the
    # checkpoint or config is not what we think it is.
    check("initial loss near ln(vocab)", 8.5 < loss.item() < 11.0,
          f"{loss.item():.3f} vs expected ~9.68")

print("\n" + "=" * 70)
print("READY — run the training cell." if ok else "NOT READY — fix the FAILs above.")
print("=" * 70)
