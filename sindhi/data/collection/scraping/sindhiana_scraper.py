import sys
import re
from bs4 import BeautifulSoup
from common.scraping.base_scraper import BaseScraper

class SindhianaScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir=output_dir, source_name="sindhiana", delay=0.5)
        self.base_url = "http://www.encyclopediasindhiana.org"
        self.start_url = "http://www.encyclopediasindhiana.org/Index.php"

    def scrape(self, max_items: int = 1000):
        self.logger.info(f"Starting Encyclopedia Sindhiana scraper for {max_items} items")
        items_scraped = 0
        
        # In a real scrape, we would iterate through articles sequentially via ID
        # e.g. article.php?Dflt=1234
        
        for article_id in range(1, max_items + 1):
            url = f"{self.base_url}/article.php?Dflt={article_id}"
            soup = self.fetch_page(url)
            
            if not soup:
                continue
                
            # The title is usually in a bold font or h1/h2
            title_tag = soup.find("strong") or soup.find("h1")
            
            # Text is usually inside div class="col-md-9" or similar main content area
            # We just grab all paragraphs
            paragraphs = soup.find_all("p")
            
            if not paragraphs:
                continue
                
            title = title_tag.get_text().strip() if title_tag else f"Article {article_id}"
            
            text_chunks = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 10]
            text = '\\n\\n'.join(text_chunks)
            
            # Basic sanity check to ensure it contains Sindhi (Arabic block) chars
            if len(text) > 50 and re.search(r'[\\u0600-\\u06FF]', text):
                item = {
                    "url": url,
                    "title": title,
                    "text": text,
                    "source": "Encyclopedia Sindhiana"
                }
                self.save_item(item)
                items_scraped += 1
                
                if items_scraped % 50 == 0:
                    self.logger.info(f"Scraped {items_scraped}/{max_items} documents")
                    
if __name__ == "__main__":
    scraper = SindhianaScraper(output_dir="sindhi/data/collection/raw/scraped")
    scraper.scrape(max_items=10) # Small test run
