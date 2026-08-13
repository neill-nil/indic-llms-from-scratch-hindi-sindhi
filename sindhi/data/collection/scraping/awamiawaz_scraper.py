import sys
import re
from bs4 import BeautifulSoup
from common.scraping.base_scraper import BaseScraper

class AwamiAwazScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir=output_dir, source_name="awamiawaz", delay=1.0)
        self.start_url = "https://awamiawaz.pk/"

    def scrape(self, max_items: int = 1000):
        self.logger.info(f"Starting Awami Awaz scraper for {max_items} items")
        items_scraped = 0
        
        soup = self.fetch_page(self.start_url)
        if not soup:
            return
            
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Look for article links (they usually have long slugs)
            if href.startswith('https://awamiawaz.pk/') and len(href) > 30 and '-' in href:
                links.add(href)
                
        self.logger.info(f"Found {len(links)} initial links to scrape")
        
        for url in list(links)[:max_items]:
            page_soup = self.fetch_page(url)
            if not page_soup:
                continue
                
            title_tag = page_soup.find("h1")
            
            # Content is usually in div with class entry-content
            content_div = page_soup.find("div", class_="entry-content")
            
            if not title_tag or not content_div:
                continue
                
            title = title_tag.get_text().strip()
            
            for tag in content_div.find_all(['script', 'style', 'div']):
                tag.decompose()
                
            text = content_div.get_text(separator='\\n').strip()
            
            if len(text) > 100 and re.search(r'[\\u0600-\\u06FF]', text):
                item = {
                    "url": url,
                    "title": title,
                    "text": text,
                    "source": "Awami Awaz"
                }
                self.save_item(item)
                items_scraped += 1
                
                if items_scraped % 10 == 0:
                    self.logger.info(f"Scraped {items_scraped} documents")
                    
            if items_scraped >= max_items:
                break

if __name__ == "__main__":
    scraper = AwamiAwazScraper(output_dir="sindhi/data/collection/raw/scraped")
    scraper.scrape(max_items=5) # Small test run
