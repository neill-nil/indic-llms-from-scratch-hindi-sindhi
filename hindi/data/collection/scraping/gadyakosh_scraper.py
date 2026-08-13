import sys
from pathlib import Path
from bs4 import BeautifulSoup
from common.scraping.base_scraper import BaseScraper

class GadyakoshScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir=output_dir, source_name="gadyakosh", delay=0.5)
        self.base_url = "http://gadyakosh.org"
        self.random_url = f"{self.base_url}/gk/Special:Random"

    def scrape(self, max_items: int = 1000):
        self.logger.info(f"Starting Gadyakosh scraper for {max_items} items")
        items_scraped = 0
        
        while items_scraped < max_items:
            soup = self.fetch_page(self.random_url)
            if not soup:
                continue
                
            title_tag = soup.find(id="firstHeading")
            content_div = soup.find(id="mw-content-text")
            
            if not title_tag or not content_div:
                continue
                
            title = title_tag.get_text().strip()
            
            # Clean up the text by removing scripts, styles, and navigation elements
            for tag in content_div.find_all(['script', 'style', 'div', 'table']):
                tag.decompose()
                
            text = content_div.get_text(separator='\\n').strip()
            
            if len(text) > 100:
                item = {
                    "url": self.random_url, # Actual URL is random but we could extract og:url
                    "title": title,
                    "text": text,
                    "source": "Gadyakosh"
                }
                self.save_item(item)
                items_scraped += 1
                
                if items_scraped % 50 == 0:
                    self.logger.info(f"Scraped {items_scraped}/{max_items} documents")

if __name__ == "__main__":
    import sys
    max_items = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    scraper = GadyakoshScraper(output_dir="hindi/data/collection/raw/scraped")
    scraper.scrape(max_items=max_items)
