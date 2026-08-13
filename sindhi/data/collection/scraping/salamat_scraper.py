import sys
import re
from bs4 import BeautifulSoup
from common.scraping.base_scraper import BaseScraper

class SalamatScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir=output_dir, source_name="sindhsalamat", delay=1.0)
        self.start_url = "https://sindhsalamat.com/"

    def scrape(self, max_items: int = 1000):
        self.logger.info(f"Starting Sindh Salamat scraper for {max_items} items")
        items_scraped = 0
        
        soup = self.fetch_page(self.start_url)
        if not soup:
            return
            
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            # XenForo thread links usually contain 'threads/'
            if 'threads/' in href:
                url = href if href.startswith('http') else f"{self.start_url}{href.lstrip('/')}"
                links.add(url)
                
        self.logger.info(f"Found {len(links)} initial thread links to scrape")
        
        for url in list(links)[:max_items]:
            page_soup = self.fetch_page(url)
            if not page_soup:
                continue
                
            title_tag = page_soup.find("h1")
            
            # XenForo message content
            messages = page_soup.find_all("div", class_="messageContent") or page_soup.find_all("article", class_="message-body")
            
            if not title_tag or not messages:
                continue
                
            title = title_tag.get_text().strip()
            
            text_chunks = [m.get_text(separator='\\n').strip() for m in messages if len(m.get_text().strip()) > 20]
            text = '\\n\\n'.join(text_chunks)
            
            if len(text) > 100 and re.search(r'[\\u0600-\\u06FF]', text):
                item = {
                    "url": url,
                    "title": title,
                    "text": text,
                    "source": "Sindh Salamat"
                }
                self.save_item(item)
                items_scraped += 1
                
                if items_scraped % 10 == 0:
                    self.logger.info(f"Scraped {items_scraped} documents")
                    
            if items_scraped >= max_items:
                break

if __name__ == "__main__":
    scraper = SalamatScraper(output_dir="sindhi/data/collection/raw/scraped")
    scraper.scrape(max_items=5) # Small test run
