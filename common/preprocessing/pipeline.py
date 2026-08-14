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
                        yield source, "\\n".join(current_doc)
                        current_doc = []
                        char_count = 0
                else:
                    if current_doc:
                        yield source, "\\n".join(current_doc)
                        current_doc = []
                        char_count = 0
            if current_doc:
                yield source, "\\n".join(current_doc)
    
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

def run_pipeline(language: str, val_ratio: float = 0.05):
    random.seed(42)
    
    raw_dir = PROJECT_ROOT / language / "data" / "collection" / "raw"
    out_dir = PROJECT_ROOT / language / "data" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"{'='*70}\\n  Preprocessing Pipeline (Streaming) — {language.upper()}\\n{'='*70}")
    
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    seen_hashes = set()
    
    stats = {
        "total_raw": 0, "passed_filters": 0, "duplicates_removed": 0, "after_dedup": 0,
        "rejected": {"length": 0, "symbol_ratio": 0, "line_structure": 0, "repetition": 0, "script": 0},
        "train_bytes": 0, "val_bytes": 0
    }
    
    start = time.time()
    with open(train_path, 'w', encoding='utf-8') as f_train, \
         open(val_path, 'w', encoding='utf-8') as f_val:
         
        for source, text in iter_raw_documents(raw_dir):
            stats["total_raw"] += 1
            
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
            
            out_str = f"{cleaned}\\n{DOC_SEPARATOR}\\n"
            out_bytes = len(out_str.encode('utf-8'))
            
            if random.random() >= val_ratio:
                f_train.write(out_str)
                stats["train_bytes"] += out_bytes
            else:
                f_val.write(out_str)
                stats["val_bytes"] += out_bytes
            
            if stats["total_raw"] % 500_000 == 0:
                logger.info(f"  ...processed {stats['total_raw']:,} raw | {stats['after_dedup']:,} clean & deduped")
                
    total_bytes = stats["train_bytes"] + stats["val_bytes"]
    stats["total_clean_bytes"] = total_bytes
    stats["estimated_clean_tokens"] = total_bytes // 10
    
    with open(out_dir / "preprocessing_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
        
    logger.info(f"Pipeline complete in {time.time()-start:.1f}s")
    logger.info(f"  train.txt: {stats['train_bytes']/1024/1024:.2f} MB")
    logger.info(f"  val.txt:   {stats['val_bytes']/1024/1024:.2f} MB")
    logger.info("DONE!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    args = parser.parse_args()
    run_pipeline(args.language)

if __name__ == "__main__":
    main()
