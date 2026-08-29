"""
Main preprocessing pipeline for Hindi and Sindhi corpora.

Reads raw .txt and .jsonl files from collection/raw/, applies all
heuristic filters, deduplicates on-the-fly, and randomly assigns
documents to train/val splits directly to avoid disk/RAM overhead.

Fully streaming and memory-efficient — runs in <300MB RAM.

Usage:
    python -m common.preprocessing.pipeline --language hindi
    python -m common.preprocessing.pipeline --language sindhi
"""

import argparse
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Generator, Tuple

from common.preprocessing.make_splits import SEED, VAL_RATIO, TEST_RATIO, assign_split
from common.preprocessing.filters import (
    length_filter,
    symbol_ratio_filter,
    line_structure_filter,
    repetition_filter,
    blocklist_filter,
    personal_data_filter,
    script_filter,
    normalize_unicode,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("preprocessing")

DOC_SEPARATOR = "<|endoftext|>"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def iter_raw_documents(raw_dir: Path) -> Generator[Tuple[str, str], None, None]:
    for txt_file in sorted(raw_dir.glob("*.txt")):
        source = txt_file.stem
        logger.info(f"  Reading {txt_file.name}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            current_doc = []
            char_count = 0
            for line in f:
                stripped = line.strip()
                if stripped:
                    current_doc.append(stripped)
                    char_count += len(stripped)
                    # Force yield if a single document exceeds ~1MB to prevent OOM
                    if char_count > 1_000_000:
                        yield source, "\n".join(current_doc)
                        current_doc = []
                        char_count = 0
                else:
                    if current_doc:
                        yield source, "\n".join(current_doc)
                        current_doc = []
                        char_count = 0
            if current_doc:
                yield source, "\n".join(current_doc)
    
    scraped_dir = raw_dir / "scraped"
    if scraped_dir.exists():
        for jsonl_file in sorted(scraped_dir.glob("*.jsonl")):
            source = jsonl_file.stem
            logger.info(f"  Reading {jsonl_file.name}...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "").strip()
                        if text: yield source, text
                    except json.JSONDecodeError:
                        continue
                        
        for txt_file in sorted(scraped_dir.glob("*.txt")):
            source = txt_file.stem
            logger.info(f"  Reading {txt_file.name}...")
            with open(txt_file, 'r', encoding='utf-8') as f:
                current_doc = []
                char_count = 0
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        current_doc.append(stripped)
                        char_count += len(stripped)
                        if char_count > 1_000_000:
                            yield source, "\n".join(current_doc)
                            current_doc = []
                            char_count = 0
                    else:
                        if current_doc:
                            yield source, "\n".join(current_doc)
                            current_doc = []
                            char_count = 0
                if current_doc:
                    yield source, "\n".join(current_doc)

# Source-name prefixes written by the download scripts (download_public_data.py,
# extract_arnizamani.py). Everything else — Wikipedia XML dump extraction and the
# targeted site scrapers — counts as manually collected.
DOWNLOADED_SOURCE_PREFIXES = (
    "sangraha", "indiccorp", "arnizamani", "sindhi_corpus",
    "sindhi_nlp_corpus", "sindhi_texts_big", "cc100", "cultuura",
)


def classify_source(source: str) -> str:
    """Label a source as 'downloaded' (bulk public corpus) or 'manual' (scraped/parsed)."""
    name = source.lower()
    if any(name.startswith(p) for p in DOWNLOADED_SOURCE_PREFIXES):
        return "downloaded"
    return "manual"


def apply_filters(text: str, language: str) -> Tuple[str, str]:
    text = normalize_unicode(text)
    text = blocklist_filter(text)
    text = personal_data_filter(text)
    
    if (res := length_filter(text)) is None: return "", "length"
    if (res := symbol_ratio_filter(res)) is None: return "", "symbol_ratio"
    if (res := line_structure_filter(res)) is None: return "", "line_structure"
    if (res := repetition_filter(res)) is None: return "", "repetition"
    if (res := script_filter(res, language)) is None: return "", "script"
    
    return res, ""

def run_pipeline(language: str, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO):
    random.seed(SEED)
    
    raw_dir = PROJECT_ROOT / language / "data" / "collection" / "raw"
    out_dir = PROJECT_ROOT / language / "data" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"{'='*70}\n  Preprocessing Pipeline (Streaming) — {language.upper()}\n{'='*70}")
    
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    test_path = out_dir / "test.txt"
    seen_hashes = set()
    
    stats = {
        "total_raw": 0, "passed_filters": 0, "duplicates_removed": 0, "after_dedup": 0,
        "rejected": {"length": 0, "symbol_ratio": 0, "line_structure": 0, "repetition": 0, "script": 0},
        "train_bytes": 0, "val_bytes": 0, "test_bytes": 0
    }

    # Per-source attribution. The brief (§1.2) requires reporting the manual vs.
    # downloaded split of the final corpus, which can only be measured while the
    # source label is still attached to each document — once everything is
    # concatenated into train.txt the provenance is unrecoverable.
    by_source = {}

    start = time.time()
    with open(train_path, 'w', encoding='utf-8') as f_train, \
         open(val_path, 'w', encoding='utf-8') as f_val, \
         open(test_path, 'w', encoding='utf-8') as f_test:
         
        for source, text in iter_raw_documents(raw_dir):
            stats["total_raw"] += 1
            src = by_source.setdefault(
                source,
                {"collection": classify_source(source), "raw_docs": 0,
                 "kept_docs": 0, "kept_bytes": 0})
            src["raw_docs"] += 1

            cleaned, reason = apply_filters(text, language)
            if reason:
                stats["rejected"][reason] += 1
                continue
            
            stats["passed_filters"] += 1
            
            h = hashlib.sha256(cleaned.encode('utf-8')).hexdigest()
            if h in seen_hashes:
                stats["duplicates_removed"] += 1
                continue
                
            seen_hashes.add(h)
            stats["after_dedup"] += 1
            
            out_str = f"{cleaned}\n{DOC_SEPARATOR}\n"
            out_bytes = len(out_str.encode('utf-8'))
            src["kept_docs"] += 1
            src["kept_bytes"] += out_bytes
            
            # Split boundaries live in make_splits.assign_split so the pipeline and the
            # documented split definition cannot drift apart.
            target = assign_split(random.random())
            {"train": f_train, "val": f_val, "test": f_test}[target].write(out_str)
            stats[f"{target}_bytes"] += out_bytes
            
            if stats["total_raw"] % 500_000 == 0:
                logger.info(f"  ...processed {stats['total_raw']:,} raw | {stats['after_dedup']:,} clean & deduped")
                
    total_bytes = stats["train_bytes"] + stats["val_bytes"] + stats["test_bytes"]
    stats["total_clean_bytes"] = total_bytes
    stats["estimated_clean_tokens"] = "Run count_tokens.py post-training for exact count"

    # Roll per-source counts up into the manual-vs-downloaded summary required by §1.2.
    stats["by_source"] = by_source
    manual_bytes = sum(v["kept_bytes"] for v in by_source.values()
                       if v["collection"] == "manual")
    stats["provenance"] = {
        "manual_bytes": manual_bytes,
        "downloaded_bytes": total_bytes - manual_bytes,
        "manual_fraction_of_clean_bytes": round(manual_bytes / total_bytes, 4) if total_bytes else 0.0,
    }
    
    with open(out_dir / "preprocessing_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
        
    logger.info(f"Pipeline complete in {time.time()-start:.1f}s")
    logger.info(f"  train.txt: {stats['train_bytes']/1024/1024:.2f} MB")
    logger.info(f"  val.txt:   {stats['val_bytes']/1024/1024:.2f} MB")
    logger.info(f"  test.txt:  {stats['test_bytes']/1024/1024:.2f} MB")
    logger.info("DONE!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    args = parser.parse_args()
    run_pipeline(args.language)

if __name__ == "__main__":
    main()
