import argparse
import bz2
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import urllib.request
import time

WIKI_URLS = {
    "hindi": "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2",
    "sindhi": "https://dumps.wikimedia.org/sdwiki/latest/sdwiki-latest-pages-articles.xml.bz2"
}

def clean_wiki_markup(text):
    if not text:
        return ""
    # Very basic Wiki markup stripping for the sake of speed
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL) # remove templates
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text) # extract text from [[Link|Text]]
    text = re.sub(r'<ref.*?</ref>', '', text, flags=re.DOTALL) # remove refs
    text = re.sub(r'<.*?>', '', text) # remove HTML tags
    text = re.sub(r'==+.*?==+', '', text) # remove headings
    text = re.sub(r"''+", "", text) # remove bold/italic marks
    # Strip some common wiki table artifacts
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL) 
    return text.strip()

import requests

def download_and_parse(language, output_dir):
    url = WIKI_URLS[language]
    filename = url.split("/")[-1]
    compressed_path = output_dir / filename
    output_path = output_dir / "scraped" / f"scraped_wiki_{language}.txt"

    # Ensure output dir exists
    (output_dir / "scraped").mkdir(parents=True, exist_ok=True)

    if not compressed_path.exists():
        print(f"Downloading {url} to {compressed_path}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(compressed_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to download: {e}")
            return
    else:
        print(f"File {compressed_path} already exists. Skipping download.")

    print(f"Extracting and parsing {compressed_path}...")
    bytes_written = 0
    doc_count = 0
    
    start_time = time.time()
    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            with bz2.BZ2File(compressed_path, "r") as bz2_file:
                # iterparse is memory efficient
                context = ET.iterparse(bz2_file, events=("end",))
                for event, elem in context:
                    # Strip namespace from tag if present: e.g. {http://www.mediawiki.org/xml/export-0.10/}text
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    if tag == "text":
                        raw_text = elem.text
                        if raw_text:
                            clean_text = clean_wiki_markup(raw_text)
                            if len(clean_text) > 100:  # Skip empty or very short snippets
                                f_out.write(clean_text + "\n\n")
                                bytes_written += len(clean_text.encode("utf-8")) + 2
                                doc_count += 1
                        elem.clear() # clear memory
    except Exception as e:
        print(f"Error parsing XML: {e}")

    elapsed = time.time() - start_time
    print(f"\nFinished parsing {language.capitalize()} Wiki.")
    print(f"Articles saved: {doc_count:,}")
    print(f"Text size: {bytes_written / 1024 / 1024:.2f} MB")
    print(f"Time taken: {elapsed:.2f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["hindi", "sindhi"], required=True)
    args = parser.parse_args()

    # Determine base directory
    base_dir = Path(__file__).resolve().parent.parent.parent
    lang_dir = base_dir / args.language / "data" / "collection" / "raw"
    lang_dir.mkdir(parents=True, exist_ok=True)

    download_and_parse(args.language, lang_dir)

if __name__ == "__main__":
    main()
