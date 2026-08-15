"""
Download Sindhi public corpus data from HuggingFace for LM pretraining.

This script targets ~400-500M tokens (~4-5 GB of raw text) of Sindhi data.
As Sindhi is a low-resource language, we combine multiple diverse sources:

  1. arnizamani/Sindhi-texts-big-dataset: Contains CC100 Sindhi, FineWeb2,
     thousands of books from Sindh Salamat, newspapers, and more.
  2. sherrycodes911/Sindhi-NLP-Corpus: A large aggregated corpus (~3.3GB JSONL).
  3. aakashMeghwar01/sindhi-corpus-505m: Contains parquet files (~850MB).
  4. ambile-official/Sindhi_Mega_Corpus_118_Million_Tokens: Contains JSON corpora.

Usage:
    python download_public_data.py                      # download all
    python download_public_data.py --dry-run             # check file sizes only
"""

import argparse
import json
import os
import sys
import time
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi, snapshot_download
from tqdm import tqdm


SOURCES = [
    {
        "id": "sindhi_texts_big_cc100",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_file",
        "filename": "CC100-Sindhi/sd.txt",
        "target_gb": 0.5,
        "desc": "CC100 Sindhi",
    },
    {
        "id": "sindhi_nlp_corpus",
        "hf_repo": "sherrycodes911/Sindhi-NLP-Corpus",
        "repo_type": "dataset",
        "format": "parquet",
        "file_pattern": "dataset/sindhi_corpus.parquet",
        "text_col": "text",
        "target_gb": 2.0,
        "desc": "Aggregated Sindhi NLP Corpus",
    },
    {
        "id": "sindhi_corpus_505m",
        "hf_repo": "aakashMeghwar01/sindhi-corpus-505m",
        "repo_type": "dataset",
        "format": "parquet",
        "file_pattern": "data/train-0000{i}-of-00005.parquet",
        "file_indices": [0, 1, 2],
        "text_col": "text",
        "desc": "Sindhi Corpus 505M (Parquet)",
    },
    {
        "id": "arnizamani_fineweb2",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_file",
        "filename": "fineweb2/fineweb2_snd.jsonl",
        "is_jsonl": True,
        "desc": "FineWeb2 Sindhi",
    },
    {
        "id": "arnizamani_books",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_snapshot",
        "allow_patterns": ["books.sindhsalamat.com/*.txt"],
        "desc": "Sindh Salamat Books",
    },
    {
        "id": "arnizamani_literature",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_snapshot",
        "allow_patterns": ["sindhiadabiboard.org/**/*.txt"],
        "desc": "Sindhi Adabi Board Literature",
    },
    {
        "id": "arnizamani_newspapers",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_snapshot",
        "allow_patterns": ["newspapers/**/*.txt"],
        "desc": "Sindhi Newspapers (Awami Awaz, Kawish, Express)",
    },
    {
        "id": "arnizamani_dictionary",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_snapshot",
        "allow_patterns": ["jamesindhilughaat/**/*.txt"],
        "desc": "Classical Sindhi Dictionary",
    },
    {
        "id": "arnizamani_blogs",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_file",
        "filename": "blogs/bhurgri.com.txt",
        "desc": "Bhurgri Blogs",
    },
    {
        "id": "arnizamani_sindhiana",
        "hf_repo": "arnizamani/Sindhi-texts-big-dataset",
        "repo_type": "dataset",
        "format": "hf_snapshot",
        "allow_patterns": ["sindhiana.org/**/*.txt"],
        "desc": "Encyclopedia Sindhiana",
    },
    {
        "id": "ambile_mega_corpus",
        "hf_repo": "ambile-official/Sindhi_Mega_Corpus_118_Million_Tokens",
        "repo_type": "dataset",
        "format": "hf_snapshot",
        "allow_patterns": ["*.json", "**/*.json"],
        "is_json": True,
        "text_key": "text",
        "desc": "Sindhi Mega Corpus 118M",
    },
]


def format_bytes(num_bytes: int) -> str:
    """Format byte count into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def estimate_tokens(num_bytes: int) -> int:
    """
    Rough estimate of BPE tokens from raw UTF-8 byte count for Sindhi.
    Sindhi (Arabic script): ~2 bytes/char, BPE token ≈ 4-5 chars → ~10 bytes/token.
    """
    return num_bytes // 10


def download_parquet_source(source: dict, output_dir: Path, dry_run: bool = False) -> dict:
    """Download parquet files from HuggingFace and extract text."""
    src_id = source["id"]
    out_file = output_dir / f"{src_id}.txt"
    stats_file = output_dir / f"{src_id}_stats.json"

    if out_file.exists() and stats_file.exists():
        with open(stats_file, "r") as f:
            existing_stats = json.load(f)
        if existing_stats.get("complete", False):
            print(f"  ✓ {src_id} already complete ({format_bytes(existing_stats['bytes_written'])}). Skipping.")
            return existing_stats

    # Handle single file or patterned files
    if "file_pattern" in source and "file_indices" in source:
        parquet_files = [source["file_pattern"].format(i=i) for i in source["file_indices"]]
    else:
        parquet_files = [source["file_pattern"]]

    print(f"\n{'='*70}")
    print(f"  Source: {source['desc']}")
    print(f"  Repo: {source['hf_repo']}")
    print(f"  Files: {len(parquet_files)} parquet files")
    print(f"  Output: {out_file}")
    print(f"{'='*70}")

    if dry_run:
        return {"source": src_id, "status": "dry_run_ok", "files": parquet_files}

    bytes_written = 0
    doc_count = 0
    start_time = time.time()
    
    target_bytes = float('inf')
    if "target_gb" in source:
        target_bytes = source["target_gb"] * 1024**3

    with open(out_file, "w", encoding="utf-8") as f_out:
        for pf in parquet_files:
            if bytes_written >= target_bytes:
                break
                
            print(f"  Downloading {pf}...")
            try:
                local_path = hf_hub_download(
                    repo_id=source["hf_repo"],
                    filename=pf,
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"    ✗ Failed to download {pf}: {e}")
                continue

            print(f"    Extracting text...")
            table = pq.read_table(local_path, columns=[source["text_col"]])
            texts = table.column(source["text_col"]).to_pylist()

            for text in tqdm(texts, desc=f"    {pf.split('/')[-1]}", ncols=80):
                if not text or len(text.strip()) < 20:
                    continue
                line = text.strip() + "\n\n"
                line_bytes = len(line.encode("utf-8"))
                f_out.write(line)
                bytes_written += line_bytes
                doc_count += 1
                
                if bytes_written >= target_bytes:
                    break

            try:
                os.remove(local_path)
            except OSError:
                pass

            print(f"    Running total: {format_bytes(bytes_written)} | {doc_count:,} docs")

    elapsed = time.time() - start_time
    stats = {
        "source": src_id,
        "status": "complete",
        "complete": True,
        "hf_repo": source["hf_repo"],
        "bytes_written": bytes_written,
        "bytes_human": format_bytes(bytes_written),
        "doc_count": doc_count,
        "estimated_tokens": estimate_tokens(bytes_written),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def download_hf_file(source: dict, output_dir: Path, dry_run: bool = False) -> dict:
    """Download a specific text file from a HuggingFace repo."""
    src_id = source["id"]
    out_file = output_dir / f"{src_id}.txt"
    stats_file = output_dir / f"{src_id}_stats.json"

    if out_file.exists() and stats_file.exists():
        with open(stats_file, "r") as f:
            existing_stats = json.load(f)
        if existing_stats.get("complete", False):
            print(f"  ✓ {src_id} already complete. Skipping.")
            return existing_stats

    print(f"\n{'='*70}")
    print(f"  Source: {source['desc']}")
    print(f"  Repo: {source['hf_repo']}")
    print(f"  File: {source['filename']}")
    print(f"  Output: {out_file}")
    print(f"{'='*70}")

    if dry_run:
        return {"source": src_id, "status": "dry_run_ok"}

    start_time = time.time()
    
    print(f"  Downloading {source['filename']}...")
    try:
        local_path = hf_hub_download(
            repo_id=source["hf_repo"],
            filename=source["filename"],
            repo_type="dataset",
        )
    except Exception as e:
        print(f"    ✗ Failed to download {source['filename']}: {e}")
        return {"source": src_id, "status": "failed"}

    print("  Copying data to output file...")
    bytes_written = 0
    doc_count = 0
    is_jsonl = source.get("is_jsonl", False)
    text_key = source.get("text_key", "text")
    
    with open(local_path, "r", encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
                
            if is_jsonl:
                try:
                    data = json.loads(line)
                    text = data.get(text_key, "")
                except json.JSONDecodeError:
                    text = ""
            else:
                text = line.strip()
                
            if not text:
                continue
                
            out_line = text.strip() + "\n\n"
            f_out.write(out_line)
            bytes_written += len(out_line.encode("utf-8"))
            doc_count += 1
            
    try:
        os.remove(local_path)
    except OSError:
        pass

    elapsed = time.time() - start_time
    stats = {
        "source": src_id,
        "status": "complete",
        "complete": True,
        "hf_repo": source["hf_repo"],
        "bytes_written": bytes_written,
        "bytes_human": format_bytes(bytes_written),
        "doc_count": doc_count,
        "estimated_tokens": estimate_tokens(bytes_written),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def download_hf_snapshot(source: dict, output_dir: Path, dry_run: bool = False) -> dict:
    """Download multiple files from a HuggingFace repo matching allow_patterns."""
    src_id = source["id"]
    out_file = output_dir / f"{src_id}.txt"
    stats_file = output_dir / f"{src_id}_stats.json"

    if out_file.exists() and stats_file.exists():
        with open(stats_file, "r") as f:
            existing_stats = json.load(f)
        if existing_stats.get("complete", False):
            print(f"  ✓ {src_id} already complete. Skipping.")
            return existing_stats

    print(f"\n{'='*70}")
    print(f"  Source: {source['desc']}")
    print(f"  Repo: {source['hf_repo']}")
    print(f"  Patterns: {source.get('allow_patterns', [])}")
    print(f"  Output: {out_file}")
    print(f"{'='*70}")

    if dry_run:
        return {"source": src_id, "status": "dry_run_ok"}

    start_time = time.time()
    
    print(f"  Downloading snapshot...")
    try:
        local_dir = snapshot_download(
            repo_id=source["hf_repo"],
            repo_type="dataset",
            allow_patterns=source.get("allow_patterns"),
            local_dir=output_dir / f"tmp_{src_id}",
        )
    except Exception as e:
        print(f"    ✗ Failed to download snapshot: {e}")
        return {"source": src_id, "status": "failed"}

    print("  Processing downloaded files into single output...")
    bytes_written = 0
    doc_count = 0
    is_json = source.get("is_json", False)
    text_key = source.get("text_key", "text")
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        for root, _, files in os.walk(local_dir):
            for file in files:
                file_path = Path(root) / file
                if not file_path.is_file():
                    continue
                    
                if file_path.suffix == ".json" or is_json:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f_in:
                            data = json.load(f_in)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and text_key in item:
                                        text = item.get(text_key, "").strip()
                                        if text:
                                            f_out.write(text + "\n\n")
                                            bytes_written += len((text + "\n\n").encode("utf-8"))
                                            doc_count += 1
                            elif isinstance(data, dict) and text_key in data:
                                text = data.get(text_key, "").strip()
                                if text:
                                    f_out.write(text + "\n\n")
                                    bytes_written += len((text + "\n\n").encode("utf-8"))
                                    doc_count += 1
                    except Exception as e:
                        # Fallback for invalid JSON or jsonl that ended up as .json
                        pass
                else:
                    # Treat as plain text
                    try:
                        with open(file_path, "r", encoding="utf-8") as f_in:
                            text = f_in.read().strip()
                            if text:
                                f_out.write(text + "\n\n")
                                bytes_written += len((text + "\n\n").encode("utf-8"))
                                doc_count += 1
                    except Exception:
                        pass

    print("  Cleaning up temp directory...")
    try:
        shutil.rmtree(local_dir)
    except OSError:
        pass

    elapsed = time.time() - start_time
    stats = {
        "source": src_id,
        "status": "complete",
        "complete": True,
        "hf_repo": source["hf_repo"],
        "bytes_written": bytes_written,
        "bytes_human": format_bytes(bytes_written),
        "doc_count": doc_count,
        "estimated_tokens": estimate_tokens(bytes_written),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download Sindhi public corpus data.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Sindhi Public Data Download")
    print(f"  Output directory: {output_dir}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'DOWNLOAD'}")
    print("=" * 70)

    all_stats = []
    for source in SOURCES:
        try:
            if source["format"] == "parquet":
                stats = download_parquet_source(source, output_dir, args.dry_run)
            elif source["format"] == "hf_file":
                stats = download_hf_file(source, output_dir, args.dry_run)
            elif source["format"] == "hf_snapshot":
                stats = download_hf_snapshot(source, output_dir, args.dry_run)
            else:
                print(f"Unknown format {source['format']} for {source['id']}")
                continue
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {source['id']}: {e}")

    print(f"\n{'='*70}\n  DOWNLOAD SUMMARY\n{'='*70}")
    total_bytes = 0
    total_tokens = 0

    for stats in all_stats:
        if stats.get("status") == "complete":
            total_bytes += stats["bytes_written"]
            total_tokens += stats["estimated_tokens"]
            print(f"  ✓ {stats['source']:25s} | {stats['bytes_human']:>10s} | ~{stats['estimated_tokens']/1e6:.1f}M tokens")

    if not args.dry_run:
        print(f"  {'─'*65}")
        print(f"  TOTAL (NEW + EXISTING): {format_bytes(total_bytes):>10s} | ~{total_tokens/1e6:.1f}M tokens")

if __name__ == "__main__":
    main()
