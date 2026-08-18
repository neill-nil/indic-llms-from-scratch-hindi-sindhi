# Sindhi Dataset Statistics

## 1. Corpus Size and Splits
* **Total Final Tokens (Counted):** 647,780,249 tokens
* **Total Final Documents:** 5,102,663
* **Data Splits (95/2.5/2.5):**
  * **Train (`train.txt`):** 3.76 GB
  * **Validation (`val.txt`):** 92 MB (423,127 lines)
  * **Test (`test.txt`):** 92 MB (423,128 lines)

## 2. Data Sources
The corpus was constructed using a massive hybrid aggregation approach to gather every possible byte of public Sindhi data:
1. **Public Datasets (Bulk):** HuggingFace Repositories (Cultuura, Wikipedia, CC100).
2. **Manual Extraction (Targeted):** Wikipedia XML Database Dumps, Sindhiana Encyclopedia Web Scraping, and custom text file parsing.

## 3. Manual Extraction Fraction
* **Required Manual Fraction:** > 20%
* **Manual Sourcing:** Because high-quality bulk datasets are extremely scarce for Sindhi, the pipeline required manual parsing. The direct parsing of the Wikipedia XML dump using a custom extractor (`scrape_wiki_dump.py`) and the manual scraping of news sources and books using custom targeted scrapers (`awamiawaz_scraper.py`, `salamat_scraper.py`, `sindhiana_scraper.py`) fulfilled the manual fraction requirement. The remaining bulk was strictly sourced from aggregated HuggingFace collections (e.g., Cultuura, CC100) via `download_public_data.py`.

## 4. Cleaning and Preprocessing Steps
The raw data (7,637,272 documents) underwent a rigorous heuristic cleaning pipeline (`pipeline.py`) that operated entirely without ML models:
1. **Length Filter:** Rejected documents shorter than 20 characters or longer than 10,000 characters.
2. **Symbol Ratio Filter:** Rejected documents where symbols/punctuation comprised more than 20% of the text.
3. **Repetition Filter:** Rejected documents with excessive character or n-gram repetition.
4. **Script Detection:** Enforced that >80% of characters must belong to the Arabic Unicode block (which covers the Sindhi Perso-Arabic alphabet).
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

**Final Yield:** 66.8% of the raw documents passed the rigorous filtering phases. The higher rejection rate (compared to Hindi's 82.2%) was expected due to the messy nature of web-scraped data and cross-language noise in low-resource corpora.
