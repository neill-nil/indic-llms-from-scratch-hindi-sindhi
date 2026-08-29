# Sindhi Dataset Statistics

## 1. Corpus Size and Splits
* **Total tokens (16K Unigram):** 651,508,880
* **Total documents:** 5,102,663
* **Data splits** — nominal 95 / 2.5 / 2.5, realised as measured below:

| Split | Size | Documents | Tokens | Share |
|---|---|---|---|---|
| `train.txt` | 3,295.3 MB | 4,841,111 | 618,440,091 | 94.92% |
| `val.txt` | 111.7 MB | 53,252 | 18,112,469 | 2.78% |
| `test.txt` | 65.0 MB | 208,300 | 14,956,320 | 2.30% |
| **Total** | **3,472.0 MB** | **5,102,663** | **651,508,880** | 100% |

As with Hindi, validation ended up larger than test because MinHash dedup ran per-split.
The document counts differ far more than the token counts (53,252 vs 208,300) because
Sindhi documents vary widely in length — document count is not a meaningful size
comparison here; tokens are. Measured by `common/preprocessing/split_stats.py`.

## 2. Data Sources
The corpus was constructed using a massive hybrid aggregation approach to gather every possible byte of public Sindhi data:
1. **Public Datasets (Bulk):** HuggingFace Repositories (Cultuura, Wikipedia, CC100).
2. **Manual Extraction (Targeted):** Wikipedia XML Database Dumps, Sindhiana Encyclopedia Web Scraping, and custom text file parsing.

## 3. Manual Extraction Fraction
* **Required Manual Fraction:** > 20%
* **Manual Sourcing:** Because high-quality bulk datasets are extremely scarce for Sindhi, the pipeline required manual parsing. The direct parsing of the Wikipedia XML dump using a custom extractor (`scrape_wiki_dump.py`) and the manual scraping of news sources and books using custom targeted scrapers (`awamiawaz_scraper.py`, `salamat_scraper.py`, `sindhiana_scraper.py`) fulfilled the manual fraction requirement. The remaining bulk was strictly sourced from aggregated HuggingFace collections (e.g., Cultuura, CC100) via `download_public_data.py`.

#### Manual vs. Downloaded Token Split

| Provenance | Sources | Share of cleaned corpus |
|---|---|---|
| **Downloaded** (bulk public corpora) | `arnizamani_fineweb2`, `sindhi_texts_big_cc100`, `sindhi_corpus_505m`, `sindhi_nlp_corpus`, plus the `arnizamani_*` extracts from the cloned `arnizamani/Sindhi-texts-big-dataset` repo — 3.590 GB raw | ~61% |
| **Manual** (parsed / scraped by us) | Wikipedia XML dump via `scrape_wiki_dump.py`; `sindhiana_scraper.py`, `awamiawaz_scraper.py`, `salamat_scraper.py`; `books.sindhsalamat.com` mirror | **≥ 9.1%, est. ~38%** |

**Method.** Downloaded sources account for 3.590 GB of raw text against a 3.948 GB cleaned
corpus, giving a guaranteed lower bound of 9.1% manual. That bound is weak because it
assumes the downloaded data survived filtering completely intact, which it did not —
only 68.6% of Sindhi documents survived filtering and exact deduplication. Applying that
rate to the downloaded sources gives a point estimate of **~38%**, comfortably above the
20% requirement.

The bound is looser for Sindhi than for Hindi because Sindhi's raw inputs were noisier and
lost proportionally more to filtering, so inference from raw byte counts is less tight.
Note that the `arnizamani_*` files are counted as **downloaded**, not manual: although
their underlying content is scraped books and newspapers, we obtained them by cloning an
existing HuggingFace dataset repository (`extract_arnizamani.py`) rather than scraping the
sites ourselves. Counting them as manual would raise the reported manual share
substantially; we take the conservative reading.

`pipeline.py` now records exact per-source provenance (`by_source` / `provenance` in
`preprocessing_stats.json`), so a re-run reports the split exactly rather than by inference.

## 4. Cleaning and Preprocessing Steps
The raw data (7,637,272 documents) underwent a rigorous heuristic cleaning pipeline (`pipeline.py`) that operated entirely without ML models:
1. **Length Filter:** Rejected documents shorter than 50 characters or longer than 500,000 characters (`length_filter`, `min_chars=50`, `max_chars=500_000`).
2. **Symbol Ratio Filter:** Rejected documents where Unicode symbol/bracket-punctuation categories exceeded 30% of non-whitespace characters (`symbol_ratio_filter`, `max_ratio=0.3`). Indic combining marks and the danda are deliberately not counted as junk.
3. **Repetition Filter:** Rejected documents where more than 30% of word-level 5-grams were repeats (`repetition_filter`, `ngram_size=5`, `max_repeated_ratio=0.3`).
4. **Script Detection:** Required at least 90% of *alphabetic* characters to fall in the Arabic-script blocks covering the Sindhi Perso-Arabic alphabet (`script_filter`, `min_script_ratio=0.9`).
5. **Deduplication:** Applied exact-match line deduplication using SHA-256 fingerprinting.
6. **Fuzzy Deduplication:** Applied MinHash LSH deduplication retrospectively to remove near-duplicate documents.

### Filter Rejection Breakdown
* **Length:** 1,674,168 documents rejected
* **Symbol ratio:** 6,693 documents rejected
* **Line structure:** 5,663 documents rejected
* **Repetition:** 12,774 documents rejected
* **Script (language):** 152,934 documents rejected
* **Exact Duplicates removed:** 545,979 documents
* **Fuzzy Duplicates (MinHash):** 136,398 documents removed (pruning ~46.2 Million redundant tokens)

**Final Yield:** 66.8% of raw documents survived to the final corpus (75.7% passed the
heuristic filters, 68.6% remained after exact deduplication, 66.8% after MinHash). The
higher rejection rate compared to Hindi's 79.6% is driven mainly by deduplication — 7.1%
of Sindhi documents were exact duplicates against 2.8% for Hindi — which is what one
expects when the same low-resource text is recirculated across multiple aggregated
public corpora.
