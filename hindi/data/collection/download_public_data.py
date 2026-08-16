"""
Download Hindi public corpus data from multiple sources for LM pretraining.

Downloads individual files from HuggingFace Hub to avoid pulling entire
datasets. Targets ~400M+ tokens (80% of 500M target) spread across 3
diverse sources (based on measured 1 parquet ≈ 140M tokens / 1.1 GB text):

  1. Sangraha Verified   (curated web + OCR + transcriptions)  — 2 files → ~2.2 GB → ~280M tokens
  2. Sangraha Unverified  (perplexity-filtered multilingual)   — 1 file  → ~1.1 GB → ~140M tokens
  3. IndicCorp v2         (monolingual web crawl)              — stream    ~0.3 GB →  ~38M tokens
                                                        Total ≈  ~3.6 GB → ~458M tokens

Estimated disk usage: ~3.6 GB of plain text.

Usage:
    python download_public_data.py                      # download all
    python download_public_data.py --dry-run             # check file sizes only
    python download_public_data.py --sources sangraha_verified  # single source
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Source configurations
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "id": "sangraha_verified",
        "hf_repo": "ai4bharat/sangraha",
        "repo_type": "dataset",
        "file_pattern": "verified/hin/data-{i}.parquet",
        "format": "parquet",
        "text_col": "text",
        "file_indices": list(range(3)),  # 3 files → ~3.3 GB text → ~420M tokens
        "desc": "Sangraha Verified (curated web, OCR, transcriptions)",
    },
    {
        "id": "sangraha_unverified",
        "hf_repo": "ai4bharat/sangraha",
        "repo_type": "dataset",
        "file_pattern": "unverified/hin/data-{i}.parquet",
        "format": "parquet",
        "text_col": "text",
        "file_indices": [0],  # 1 file → ~1.1 GB text → ~140M tokens
        "desc": "Sangraha Unverified (perplexity-filtered)",
    },
    {
        "id": "indiccorp_v2",
        "hf_repo": "ai4bharat/IndicCorpV2",
        "repo_type": "dataset",
        "files": ["data/hi-1.txt"],  # stream first 0.1 GB from 25 GB file
        "format": "text",
        "target_gb": 0.7,  # ~85M tokens — pushed up to ensure we clear 450M total
        "desc": "IndicCorp v2 (monolingual web crawl)",
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
    Rough estimate of BPE tokens from raw UTF-8 byte count for Hindi.
    Hindi Devanagari: ~3 bytes/char, BPE token ≈ 2-3 chars → ~6-9 bytes/token.
    We use 8 bytes/token as a middle estimate.
    """
    return num_bytes // 8


def download_parquet_source(source: dict, output_dir: Path, dry_run: bool = False) -> dict:
    """Download parquet files from HuggingFace and extract text."""
    src_id = source["id"]
    out_file = output_dir / f"{src_id}.txt"
    stats_file = output_dir / f"{src_id}_stats.json"

    # Check if already downloaded (resume support)
    if out_file.exists() and stats_file.exists():
        with open(stats_file, "r") as f:
            existing_stats = json.load(f)
        if existing_stats.get("complete", False):
            print(f"  ✓ {src_id} already complete ({format_bytes(existing_stats['bytes_written'])}). Skipping.")
            return existing_stats

    file_indices = source["file_indices"]
    parquet_files = [source["file_pattern"].format(i=i) for i in file_indices]

    print(f"\n{'='*70}")
    print(f"  Source: {source['desc']}")
    print(f"  Repo: {source['hf_repo']}")
    print(f"  Files: {len(parquet_files)} parquet files (indices {file_indices[0]}-{file_indices[-1]})")
    print(f"  Output: {out_file}")
    print(f"{'='*70}")

    if dry_run:
        api = HfApi()
        total_size = 0
        for pf in parquet_files:
            try:
                info = api.repo_info(source["hf_repo"], repo_type="dataset")
                # Get file info
                siblings = [s for s in info.siblings if s.rfilename == pf]
                if siblings:
                    size = siblings[0].size
                    total_size += size
                    print(f"  {pf:50s} {size/1024/1024:.1f} MB")
            except Exception as e:
                print(f"  ✗ {pf}: {e}")
        print(f"  Total parquet size: {format_bytes(total_size)}")
        print(f"  Estimated text size: ~{format_bytes(int(total_size * 1.0))} (roughly 1:1 for parquet text)")
        return {"source": src_id, "status": "dry_run_ok", "parquet_bytes": total_size}

    bytes_written = 0
    doc_count = 0
    empty_docs = 0
    start_time = time.time()

    with open(out_file, "w", encoding="utf-8") as f_out:
        for pf in parquet_files:
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
                if not text or len(text.strip()) < 50:
                    empty_docs += 1
                    continue
                text = text.strip()
                line = text + "\n\n"
                line_bytes = len(line.encode("utf-8"))
                f_out.write(line)
                bytes_written += line_bytes
                doc_count += 1

            # Clean up downloaded parquet to save disk space
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
        "files_downloaded": parquet_files,
        "bytes_written": bytes_written,
        "bytes_human": format_bytes(bytes_written),
        "doc_count": doc_count,
        "empty_docs_skipped": empty_docs,
        "estimated_tokens": estimate_tokens(bytes_written),
        "elapsed_seconds": round(elapsed, 1),
        "output_file": str(out_file),
    }

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  ✓ Done: {format_bytes(bytes_written)} | "
          f"{doc_count:,} docs | "
          f"~{estimate_tokens(bytes_written)/1e6:.0f}M tokens | "
          f"{elapsed:.0f}s")

    return stats


def download_text_source(source: dict, output_dir: Path, dry_run: bool = False) -> dict:
    """Stream a plain text file from HuggingFace via HTTP, capping at target_gb.

    Uses HTTP streaming to avoid downloading the full file (e.g. 25 GB)
    when we only need a fraction of it.
    """
    import requests

    src_id = source["id"]
    out_file = output_dir / f"{src_id}.txt"
    stats_file = output_dir / f"{src_id}_stats.json"
    target_bytes = int(source["target_gb"] * 1024**3)

    # Check if already downloaded (resume support)
    if out_file.exists() and stats_file.exists():
        with open(stats_file, "r") as f:
            existing_stats = json.load(f)
        if existing_stats.get("complete", False):
            print(f"  ✓ {src_id} already complete ({format_bytes(existing_stats['bytes_written'])}). Skipping.")
            return existing_stats

    print(f"\n{'='*70}")
    print(f"  Source: {source['desc']}")
    print(f"  Repo: {source['hf_repo']}")
    print(f"  Files: {source['files']}")
    print(f"  Target: {source['target_gb']} GB ({format_bytes(target_bytes)})")
    print(f"  Output: {out_file}")
    print(f"{'='*70}")

    if dry_run:
        api = HfApi()
        info = api.repo_info(source["hf_repo"], repo_type="dataset")
        for fname in source["files"]:
            siblings = [s for s in info.siblings if s.rfilename == fname]
            if siblings:
                print(f"  {fname}: {format_bytes(siblings[0].size)} (will stream only {source['target_gb']} GB)")
        return {"source": src_id, "status": "dry_run_ok"}

    start_time = time.time()
    bytes_written = 0
    doc_count = 0

    for fname in source["files"]:
        url = f"https://huggingface.co/datasets/{source['hf_repo']}/resolve/main/{fname}"
        print(f"  Streaming {fname} (stopping at {source['target_gb']} GB)...")

        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        pbar = tqdm(total=target_bytes, unit="B", unit_scale=True,
                    desc=f"    {fname}", ncols=80)

        # Buffer for handling line boundaries across chunks
        buffer = ""
        done = False

        with open(out_file, "w", encoding="utf-8") as f_out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024, decode_unicode=False):
                if done:
                    break

                # Decode chunk, handling partial UTF-8 sequences
                text = chunk.decode("utf-8", errors="replace")
                buffer += text

                # Process complete lines
                lines = buffer.split("\n")
                buffer = lines[-1]  # keep incomplete last line in buffer

                for line in lines[:-1]:
                    line = line.strip()
                    if not line or len(line) < 50:
                        continue
                    out_line = line + "\n\n"
                    out_bytes = len(out_line.encode("utf-8"))
                    f_out.write(out_line)
                    bytes_written += out_bytes
                    doc_count += 1
                    pbar.update(out_bytes)

                    if bytes_written >= target_bytes:
                        done = True
                        break

        resp.close()
        pbar.close()

    elapsed = time.time() - start_time

    stats = {
        "source": src_id,
        "status": "complete",
        "complete": True,
        "hf_repo": source["hf_repo"],
        "files": source["files"],
        "bytes_written": bytes_written,
        "bytes_human": format_bytes(bytes_written),
        "doc_count": doc_count,
        "estimated_tokens": estimate_tokens(bytes_written),
        "elapsed_seconds": round(elapsed, 1),
        "output_file": str(out_file),
    }

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  ✓ Done: {format_bytes(bytes_written)} | "
          f"{doc_count:,} docs | "
          f"~{estimate_tokens(bytes_written)/1e6:.0f}M tokens | "
          f"{elapsed:.0f}s")

    return stats


def download_source(source: dict, output_dir: Path, dry_run: bool = False) -> dict:
    """Dispatch to the correct download function based on source format."""
    if source["format"] == "parquet":
        return download_parquet_source(source, output_dir, dry_run)
    elif source["format"] == "text":
        return download_text_source(source, output_dir, dry_run)
    else:
        print(f"  ✗ Unknown format: {source['format']}")
        return {"source": source["id"], "status": "error", "error": "unknown format"}


def main():
    parser = argparse.ArgumentParser(
        description="Download Hindi public corpus data for LM pretraining."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save downloaded text files. Default: hindi/data/collection/raw/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only check file sizes without downloading.",
    )
    parser.add_argument(
        "--sources", nargs="+",
        choices=[s["id"] for s in SOURCES], default=None,
        help="Download only specific sources. Default: all.",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter sources if specified
    sources = SOURCES
    if args.sources:
        sources = [s for s in SOURCES if s["id"] in args.sources]

    print("=" * 70)
    print("  Hindi Public Data Download")
    print(f"  Output directory: {output_dir}")
    print(f"  Sources: {len(sources)}")
    print(f"  Estimated total text: ~3.5 GB  (~400-500M tokens)")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'DOWNLOAD'}")
    print("=" * 70)

    all_stats = []
    for source in sources:
        stats = download_source(source, output_dir, dry_run=args.dry_run)
        all_stats.append(stats)

    # Print summary
    print(f"\n{'='*70}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*70}")

    total_bytes = 0
    total_docs = 0
    total_tokens = 0

    for stats in all_stats:
        status = stats.get("status", "unknown")
        if status == "complete":
            total_bytes += stats["bytes_written"]
            total_docs += stats["doc_count"]
            total_tokens += stats["estimated_tokens"]
            print(f"  ✓ {stats['source']:25s} | {stats['bytes_human']:>10s} | "
                  f"{stats['doc_count']:>10,} docs | ~{stats['estimated_tokens']/1e6:.0f}M tokens")
        elif status == "dry_run_ok":
            print(f"  ○ {stats['source']:25s} | dry run OK")
        else:
            print(f"  ✗ {stats['source']:25s} | {status}: {stats.get('error', 'unknown')}")

    if total_bytes > 0:
        print(f"  {'─'*65}")
        print(f"  TOTAL: {format_bytes(total_bytes):>10s} | {total_docs:>10,} docs | ~{total_tokens/1e6:.0f}M tokens")
        print(f"\n  Files saved to: {output_dir}/")

    # Save overall summary
    summary_file = output_dir / "download_summary.json"
    summary = {
        "total_bytes": total_bytes,
        "total_bytes_human": format_bytes(total_bytes),
        "total_docs": total_docs,
        "total_estimated_tokens": total_tokens,
        "sources": all_stats,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
