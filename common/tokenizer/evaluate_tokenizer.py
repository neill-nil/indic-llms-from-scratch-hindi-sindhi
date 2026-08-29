"""
Tokenizer Evaluation Script for Hindi and Sindhi Models.

Compares BPE vs Unigram tokenizers on standard metrics:
  1. Vocabulary composition analysis (script mix, byte-fallback, token lengths)
  2. Compression: fertility, tokens/char, bytes/token
  3. Token-frequency statistics: entropy, vocabulary utilisation, coverage ranks
  4. Coverage (unknown token rate)
  5. Sample tokenizations, plus automatically-mined contrastive examples where
     the two algorithms genuinely disagree

Usage:
    python -m common.tokenizer.evaluate_tokenizer \
        --language hindi --vocab_size 16000
"""

import argparse
import itertools
import json
import logging
import math
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

        # Byte-fallback tokens (e.g. <0x41>) MUST be tested before the generic
        # "<...>" special-token test, because they also match that pattern.
        # Testing in the other order silently folds all 256 byte tokens into
        # the special-token count and reports byte_fallback_tokens = 0.
        if piece.startswith("<0x") and piece.endswith(">"):
            byte_tokens += 1
            continue

        # Genuine control/special tokens: <pad> <unk> <s> </s> <|endoftext|>
        if piece.startswith("<") and piece.endswith(">"):
            special_tokens += 1
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


def iter_sampled_lines(text_path: Path, stride: int = 17, max_lines: int = 120_000):
    """
    Yield every `stride`-th non-empty content line of a split file.

    Reading a contiguous head-of-file block (e.g. the first 50k lines) biases the
    sample towards whichever source happened to be written first, so all held-out
    metrics below are computed on a strided sample spread across the whole file.
    """
    kept = 0
    with open(text_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            stripped = line.strip()
            if not stripped or stripped == DOC_SEPARATOR:
                continue
            yield stripped
            kept += 1
            if kept >= max_lines:
                return


def compute_corpus_metrics(sp: spm.SentencePieceProcessor, text_path: Path):
    """
    Encode a held-out sample and compute the tokenizer's intrinsic quality metrics.

    Reports three different compression views because they answer different questions:
      * fertility   - tokens per whitespace word. The conventional number, but it is
                      sensitive to how a language delimits words.
      * tokens/char - script-independent compression. This is the fair way to compare
                      Devanagari against Perso-Arabic, whose word lengths differ.
      * bytes/token - how much raw UTF-8 one token buys, which is what actually sets
                      how much text fits in the model's context window.

    Also returns the token-frequency statistics required by the brief (§1.3):
      * vocab utilisation - how many vocabulary slots the held-out text ever uses;
                            unused slots are wasted embedding parameters.
      * token entropy     - empirical entropy (bits/token) of the token distribution.
                            A lower-entropy token stream is intrinsically easier to
                            predict, so it lower-bounds the perplexity a downstream
                            LM can reach.
    """
    unk_id = sp.unk_id()
    total_tokens = total_words = total_chars = total_bytes = 0
    unk_count = lines_sampled = 0
    freq = Counter()

    for stripped in iter_sampled_lines(text_path):
        lines_sampled += 1
        total_words += len(stripped.split())
        total_chars += len(stripped)
        total_bytes += len(stripped.encode('utf-8'))

        pieces = sp.encode(stripped, out_type=int)
        total_tokens += len(pieces)
        freq.update(pieces)
        unk_count += sum(1 for p in pieces if p == unk_id)

    vocab_size = sp.get_piece_size()
    entropy = -sum((c / total_tokens) * math.log2(c / total_tokens)
                   for c in freq.values()) if total_tokens else 0.0

    # Frequency profile: how concentrated is probability mass over the vocabulary?
    ranked = [c for _, c in freq.most_common()]
    cumulative = list(itertools.accumulate(ranked))

    def coverage_rank(fraction: float) -> int:
        """Number of distinct token types needed to cover `fraction` of all tokens."""
        target = fraction * total_tokens
        return next((i + 1 for i, c in enumerate(cumulative) if c >= target), len(ranked))

    return {
        "fertility": round(total_tokens / max(total_words, 1), 4),
        "tokens_per_char": round(total_tokens / max(total_chars, 1), 4),
        "bytes_per_token": round(total_bytes / max(total_tokens, 1), 4),
        "chars_per_token": round(total_chars / max(total_tokens, 1), 4),
        "unk_rate": round(unk_count / max(total_tokens, 1), 8),
        "token_entropy_bits": round(entropy, 4),
        "vocab_types_used": len(freq),
        "vocab_utilisation": round(len(freq) / vocab_size, 4),
        "types_covering_50pct": coverage_rank(0.50),
        "types_covering_90pct": coverage_rank(0.90),
        "types_covering_99pct": coverage_rank(0.99),
        "lines_sampled": lines_sampled,
        "total_words_sampled": total_words,
        "total_tokens": total_tokens,
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


def mine_contrastive_examples(sp_a: spm.SentencePieceProcessor,
                              sp_b: spm.SentencePieceProcessor,
                              text_path: Path, name_a: str, name_b: str,
                              min_freq: int = 50, top_k: int = 10):
    """
    Find frequent words that `sp_a` tokenizes into strictly fewer pieces than `sp_b`.

    Report examples must be mined from the actual models rather than written by hand.
    An earlier revision of the report claimed Unigram kept a word intact that both
    tokenizers in fact split identically; generating the examples makes that class
    of error impossible.
    """
    word_freq = Counter()
    for line in iter_sampled_lines(text_path, stride=1, max_lines=60_000):
        word_freq.update(line.split())

    wins = []
    for word, count in word_freq.items():
        if count < min_freq or len(word) < 4:
            continue
        pieces_a = sp_a.encode(word, out_type=str)
        pieces_b = sp_b.encode(word, out_type=str)
        if len(pieces_a) < len(pieces_b):
            wins.append({
                "word": word,
                "corpus_frequency": count,
                name_a: pieces_a,
                name_b: pieces_b,
            })

    wins.sort(key=lambda w: -w["corpus_frequency"])
    return {"n_words_favouring_" + name_a: len(wins), "examples": wins[:top_k]}


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
    
    # 2. Held-out compression + token-frequency statistics
    fertility = compute_corpus_metrics(sp, val_path)
    logger.info(f"  Fertility:           {fertility['fertility']:.4f} tokens/word")
    logger.info(f"  Tokens per char:     {fertility['tokens_per_char']:.4f}")
    logger.info(f"  Bytes per token:     {fertility['bytes_per_token']:.3f}")
    logger.info(f"  Token entropy:       {fertility['token_entropy_bits']:.2f} bits/token")
    logger.info(f"  Vocab utilisation:   {fertility['vocab_utilisation']:.2%} "
                f"({fertility['vocab_types_used']:,} types seen)")
    logger.info(f"  UNK rate:            {fertility['unk_rate']:.8f}")
    
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
    
    both_models = "bpe" in results and "unigram" in results

    # Mine contrastive examples in both directions so neither algorithm is
    # given a free pass — each gets to show the words it handles better.
    if both_models:
        sp_bpe = spm.SentencePieceProcessor(
            model_file=str(tokenizer_dir / f"{args.language}_bpe_{args.vocab_size}.model"))
        sp_uni = spm.SentencePieceProcessor(
            model_file=str(tokenizer_dir / f"{args.language}_unigram_{args.vocab_size}.model"))
        results["contrastive"] = {
            "unigram_better": mine_contrastive_examples(
                sp_uni, sp_bpe, val_path, "unigram", "bpe"),
            "bpe_better": mine_contrastive_examples(
                sp_bpe, sp_uni, val_path, "bpe", "unigram"),
        }

    # Save comparison report
    report_path = tokenizer_dir / "evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFull report saved to: {report_path}")
    
    # Print comparison summary
    if both_models:
        logger.info("\n=== COMPARISON SUMMARY ===")
        # For every metric below, lower is better.
        for metric in ["fertility", "tokens_per_char", "token_entropy_bits", "unk_rate"]:
            bpe_val = results["bpe"]["fertility"][metric]
            uni_val = results["unigram"]["fertility"][metric]
            if bpe_val == uni_val:
                winner = "tie"
            else:
                winner = "BPE" if bpe_val < uni_val else "Unigram"
            delta = abs(bpe_val - uni_val) / max(bpe_val, 1e-12)
            logger.info(f"  {metric:>20s}: BPE={bpe_val:<10.4f} Unigram={uni_val:<10.4f} "
                        f"→ {winner} (by {delta:.2%})")

        # Higher is better: more of the vocabulary budget is actually earning its keep.
        bpe_u = results["bpe"]["fertility"]["vocab_utilisation"]
        uni_u = results["unigram"]["fertility"]["vocab_utilisation"]
        logger.info(f"  {'vocab_utilisation':>20s}: BPE={bpe_u:<10.2%} Unigram={uni_u:<10.2%} "
                    f"→ {'Unigram' if uni_u > bpe_u else 'BPE'}")
        
        bpe_ratio = results["bpe"]["vocabulary"]["target_script_ratio"]
        uni_ratio = results["unigram"]["vocabulary"]["target_script_ratio"]
        winner = "BPE" if bpe_ratio > uni_ratio else "Unigram"
        logger.info(f"  {'target_script_ratio':>20s}: BPE={bpe_ratio:.2%}  Unigram={uni_ratio:.2%}  → {winner}")
    
    # Print sample tokenizations
    for model_type, data in results.items():
        if model_type == "contrastive":
            continue
        logger.info(f"\n--- {model_type.upper()} Sample Tokenizations ---")
        for s in data.get("samples", []):
            logger.info(f"  Original:  {s['original'][:80]}...")
            logger.info(f"  Tokens({s['n_tokens']:3d}): {' | '.join(s['tokens'][:15])}")
            logger.info("")


if __name__ == "__main__":
    main()
