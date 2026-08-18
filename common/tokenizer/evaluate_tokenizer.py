"""
Tokenizer Evaluation Script for Hindi and Sindhi Models.

Compares BPE vs Unigram tokenizers on standard metrics:
  1. Vocabulary composition analysis
  2. Fertility (tokens-per-word ratio)
  3. Token length distribution
  4. Coverage (unknown token rate)
  5. Sample tokenization examples

Usage:
    python -m common.tokenizer.evaluate_tokenizer \
        --language hindi --vocab_size 16000
"""

import argparse
import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tokenizer_eval")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"

# Unicode script detection helpers
DEVANAGARI_RANGE = range(0x0900, 0x0980)
ARABIC_RANGE = range(0x0600, 0x0700)
ARABIC_EXT_RANGE = range(0x0750, 0x0780)
LATIN_RANGE = range(0x0000, 0x0080)


def detect_script(char: str) -> str:
    """Classify a character by Unicode script family."""
    cp = ord(char)
    if cp in DEVANAGARI_RANGE:
        return "devanagari"
    if cp in ARABIC_RANGE or cp in ARABIC_EXT_RANGE:
        return "arabic"
    if cp in LATIN_RANGE:
        return "latin"
    cat = unicodedata.category(char)
    if cat.startswith("N"):
        return "digit"
    if cat.startswith("P") or cat.startswith("S"):
        return "symbol"
    return "other"


def analyze_vocabulary(sp: spm.SentencePieceProcessor, language: str):
    """Analyze the composition of the learned vocabulary."""
    vocab_size = sp.get_piece_size()
    script_counts = Counter()
    token_lengths = []
    byte_tokens = 0
    special_tokens = 0
    
    target_script = "devanagari" if language == "hindi" else "arabic"
    
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        
        # Special tokens
        if piece.startswith("<") and piece.endswith(">"):
            special_tokens += 1
            continue
        
        # Byte fallback tokens (e.g., <0x41>)
        if piece.startswith("<0x") and piece.endswith(">"):
            byte_tokens += 1
            continue
        
        # Remove the SentencePiece continuation marker ▁
        clean = piece.replace("▁", "")
        if not clean:
            continue
        
        token_lengths.append(len(clean))
        
        # Classify by majority script
        char_scripts = [detect_script(c) for c in clean]
        majority = Counter(char_scripts).most_common(1)[0][0]
        script_counts[majority] += 1
    
    results = {
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "byte_fallback_tokens": byte_tokens,
        "script_distribution": dict(script_counts.most_common()),
        "target_script_ratio": script_counts[target_script] / max(sum(script_counts.values()), 1),
        "avg_token_length_chars": sum(token_lengths) / max(len(token_lengths), 1),
        "median_token_length": sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0,
    }
    
    return results


def compute_fertility(sp: spm.SentencePieceProcessor, text_path: Path,
                      max_lines: int = 50000):
    """
    Compute fertility = avg tokens per whitespace-delimited word.
    Lower fertility means the tokenizer is more efficient.
    """
    total_tokens = 0
    total_words = 0
    unk_id = sp.unk_id()
    unk_count = 0
    total_pieces = 0
    
    with open(text_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            stripped = line.strip()
            if not stripped or stripped == DOC_SEPARATOR:
                continue
            
            words = stripped.split()
            total_words += len(words)
            
            pieces = sp.encode(stripped, out_type=int)
            total_tokens += len(pieces)
            total_pieces += len(pieces)
            unk_count += sum(1 for p in pieces if p == unk_id)
    
    fertility = total_tokens / max(total_words, 1)
    unk_rate = unk_count / max(total_pieces, 1)
    
    return {
        "fertility": round(fertility, 4),
        "total_words_sampled": total_words,
        "total_tokens": total_tokens,
        "unk_rate": round(unk_rate, 6),
        "lines_sampled": min(i + 1, max_lines) if total_words > 0 else 0,
    }


def show_sample_tokenizations(sp: spm.SentencePieceProcessor, text_path: Path,
                               n_samples: int = 10):
    """Display sample tokenizations for qualitative inspection."""
    samples = []
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped == DOC_SEPARATOR:
                continue
            if len(stripped) > 20 and len(stripped) < 200:
                pieces = sp.encode(stripped, out_type=str)
                samples.append({
                    "original": stripped,
                    "tokens": pieces,
                    "n_tokens": len(pieces),
                })
            if len(samples) >= n_samples:
                break
    return samples


def evaluate_model(model_path: Path, language: str, val_path: Path):
    """Run full evaluation on a single model."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    
    logger.info(f"Evaluating: {model_path.name}")
    logger.info(f"  Vocab size: {sp.get_piece_size():,}")
    
    # 1. Vocabulary analysis
    vocab_analysis = analyze_vocabulary(sp, language)
    logger.info(f"  Script distribution: {vocab_analysis['script_distribution']}")
    logger.info(f"  Target script ratio: {vocab_analysis['target_script_ratio']:.2%}")
    logger.info(f"  Avg token length:    {vocab_analysis['avg_token_length_chars']:.2f} chars")
    
    # 2. Fertility
    fertility = compute_fertility(sp, val_path)
    logger.info(f"  Fertility:           {fertility['fertility']:.4f} tokens/word")
    logger.info(f"  UNK rate:            {fertility['unk_rate']:.6f}")
    
    # 3. Samples
    samples = show_sample_tokenizations(sp, val_path, n_samples=5)
    
    return {
        "model": model_path.name,
        "vocabulary": vocab_analysis,
        "fertility": fertility,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare tokenizer models")
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"],
                        help="Target language")
    parser.add_argument("--vocab_size", type=int, default=16000,
                        help="Vocab size to look for")
    args = parser.parse_args()
    
    tokenizer_dir = PROJECT_ROOT / args.language / "tokenizer"
    val_path = PROJECT_ROOT / args.language / "data" / "preprocessed" / "val.txt"
    
    if not val_path.exists():
        logger.error(f"Validation data not found: {val_path}")
        return
    
    results = {}
    
    for model_type in ["bpe", "unigram"]:
        model_file = tokenizer_dir / f"{args.language}_{model_type}_{args.vocab_size}.model"
        if not model_file.exists():
            logger.warning(f"  Model not found: {model_file}")
            continue
        results[model_type] = evaluate_model(model_file, args.language, val_path)
    
    if not results:
        logger.error("No models found to evaluate.")
        return
    
    # Save comparison report
    report_path = tokenizer_dir / "evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFull report saved to: {report_path}")
    
    # Print comparison summary
    if len(results) == 2:
        logger.info("\n=== COMPARISON SUMMARY ===")
        for metric in ["fertility", "unk_rate"]:
            bpe_val = results["bpe"]["fertility"][metric]
            uni_val = results["unigram"]["fertility"][metric]
            winner = "BPE" if bpe_val < uni_val else "Unigram"
            logger.info(f"  {metric:>20s}: BPE={bpe_val:.4f}  Unigram={uni_val:.4f}  → {winner}")
        
        bpe_ratio = results["bpe"]["vocabulary"]["target_script_ratio"]
        uni_ratio = results["unigram"]["vocabulary"]["target_script_ratio"]
        winner = "BPE" if bpe_ratio > uni_ratio else "Unigram"
        logger.info(f"  {'target_script_ratio':>20s}: BPE={bpe_ratio:.2%}  Unigram={uni_ratio:.2%}  → {winner}")
    
    # Print sample tokenizations
    for model_type, data in results.items():
        logger.info(f"\n--- {model_type.upper()} Sample Tokenizations ---")
        for s in data.get("samples", []):
            logger.info(f"  Original:  {s['original'][:80]}...")
            logger.info(f"  Tokens({s['n_tokens']:3d}): {' | '.join(s['tokens'][:15])}")
            logger.info("")


if __name__ == "__main__":
    main()
