# Hindi Dataset Statistics

## 1. Corpus Size and Splits
* **Total tokens (16K Unigram):** 531,471,043
* **Total documents:** 2,214,401
* **Data splits** — nominal 95 / 2.5 / 2.5, realised as measured below:

| Split | Size | Documents | Tokens | Share |
|---|---|---|---|---|
| `train.txt` | 4,720.1 MB | 2,103,125 | 504,818,280 | 94.99% |
| `val.txt` | 141.0 MB | 61,762 | 14,826,428 | 2.79% |
| `test.txt` | 108.2 MB | 49,514 | 11,826,335 | 2.23% |
| **Total** | **4,969.3 MB** | **2,214,401** | **531,471,043** | 100% |

The validation split is ~25% larger than test despite an equal nominal allocation. The
pipeline's random assignment is symmetric; the asymmetry was introduced afterwards, when
the MinHash fuzzy-dedup pass was run independently on each eval split and happened to
prune more near-duplicates from test. Both splits remain far larger than needed for stable
perplexity estimation, so this was left as-is rather than re-drawn. Measured by
`common/preprocessing/split_stats.py`; token counts encode content lines and add one
`<|endoftext|>` token per document boundary, matching what Phase 2 feeds the model.

## 2. Data Sources
The corpus was constructed using a hybrid approach to satisfy the manual extraction requirement:
1. **Public Dataset (Bulk):** IndicCorp (HuggingFace)
2. **Manual Extraction (Targeted):** Wikipedia XML Database Dumps

## 3. Manual Extraction Fraction
* **Required Manual Fraction:** > 20%
* **Manual Sourcing:** The manual extraction requirement was fulfilled by writing custom parsing scripts rather than solely relying on pre-packaged datasets. Specifically, the massive Wikipedia XML database dump was manually parsed using a custom streaming BZ2 XML extractor (`scrape_wiki_dump.py`). This, along with custom targeted scrapers (`gadyakosh_scraper.py`, `bhaskar_scraper.py`, `vikaspedia_scraper.py`), ensured we fulfilled the manual data collection component before aggregating with IndicCorp to reach the final 517M token count.

#### Manual vs. Downloaded Token Split

| Provenance | Sources | Share of cleaned corpus |
|---|---|---|
| **Downloaded** (bulk public corpora) | `sangraha_verified`, `sangraha_unverified` (AI4Bharat Sangraha), `indiccorp_v2` (AI4Bharat IndicCorp V2) — 3.692 GB raw | ~58% |
| **Manual** (parsed / scraped by us) | Wikipedia XML dump via `scrape_wiki_dump.py`; `gadyakosh_scraper.py`, `bhaskar_scraper.py`, `vikaspedia_scraper.py` | **≥ 29.4%, est. ~43%** |

**Method.** The download scripts record `bytes_written` per source, giving 3.692 GB of
downloaded raw text; the cleaned corpus is 5.231 GB. Filtering only ever removes bytes, so
the downloaded portion can contribute at most 3.692 GB of the cleaned corpus, which puts a
**guaranteed lower bound of 29.4%** on the manually-collected share. Applying the
corpus-wide 80.7% survival rate (post-filter, post-exact-dedup) to the downloaded sources
gives a point estimate of **~43%**. Either figure clears the 20% requirement.

This is a byte-share rather than a token-share. The two are near-equivalent here because
all sources are the same script and are tokenized by the same 16K Unigram model, so bytes
per token barely varies across sources. `pipeline.py` now records exact per-source
document and byte counts (`by_source` / `provenance` in `preprocessing_stats.json`), so a
re-run reports the split exactly rather than by inference.

## 4. Cleaning and Preprocessing Steps
The raw data (2,783,013 documents) underwent a rigorous heuristic cleaning pipeline (`pipeline.py`) that operated entirely without ML models:
1. **Length Filter:** Rejected documents shorter than 50 characters or longer than 500,000 characters (`length_filter`, `min_chars=50`, `max_chars=500_000`).
2. **Symbol Ratio Filter:** Rejected documents where Unicode symbol/bracket-punctuation categories exceeded 30% of non-whitespace characters (`symbol_ratio_filter`, `max_ratio=0.3`). Indic combining marks and the danda are deliberately not counted as junk.
3. **Repetition Filter:** Rejected documents where more than 30% of word-level 5-grams were repeats (`repetition_filter`, `ngram_size=5`, `max_repeated_ratio=0.3`).
4. **Script Detection:** Required at least 90% of *alphabetic* characters to fall in the Devanagari block U+0900–U+097F (`script_filter`, `min_script_ratio=0.9`).
5. **Deduplication:** Applied exact-match line deduplication using SHA-256 fingerprinting.
6. **Fuzzy Deduplication:** Applied MinHash LSH deduplication retrospectively to remove near-duplicate documents.

### Filter Rejection Breakdown
* **Length:** 290,735 documents rejected
* **Symbol ratio:** 6,253 documents rejected
* **Line structure:** 10,167 documents rejected
* **Repetition:** 1,553 documents rejected
* **Script (language):** 152,389 documents rejected
* **Exact Duplicates removed:** 76,787 documents
* **Fuzzy Duplicates (MinHash):** 30,728 documents

**Final Yield:** 79.6% of raw documents survived to the final corpus (83.4% passed the
heuristic filters, 80.7% remained after exact SHA-256 deduplication, 79.6% after the
MinHash fuzzy-dedup pass).
