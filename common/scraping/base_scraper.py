import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, Optional
import requests
from bs4 import BeautifulSoup


class BaseScraper(ABC):
    """Base class for all web scrapers in the pipeline."""

    def __init__(self, output_dir: str, source_name: str, delay: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_name = source_name
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.source_name)

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page with rate limiting and error handling."""
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def save_item(self, item: Dict):
        """Save a single scraped item to a JSONL file."""
        if not item.get("text") or len(item["text"].strip()) < 50:
            return  # Skip empty or very short texts
            
        output_file = self.output_dir / f"{self.source_name}.jsonl"
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @abstractmethod
    def scrape(self, max_items: int = 1000):
        """Main scraping loop to be implemented by subclasses."""
        pass
