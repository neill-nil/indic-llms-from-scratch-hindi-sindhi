import sys
from bs4 import BeautifulSoup
from common.scraping.base_scraper import BaseScraper

class BhaskarScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir=output_dir, source_name="bhaskar", delay=1.0)
        self.start_url = "https://www.bhaskar.com/"

    def scrape(self, max_items: int = 1000):
        self.logger.info(f"Starting Bhaskar scraper for {max_items} items")
        items_scraped = 0
        
        soup = self.fetch_page(self.start_url)
        if not soup:
            return
            
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Bhaskar article URLs typically have long slugs with IDs
            if href.startswith('/') and len(href) > 20 and '-' in href:
                links.add(f"https://www.bhaskar.com{href}")
                
        self.logger.info(f"Found {len(links)} initial links to scrape")
        
        for url in list(links)[:max_items]:
            page_soup = self.fetch_page(url)
            if not page_soup:
                continue
                
            title_tag = page_soup.find("h1")
            
            # Bhaskar wraps content in multiple divs, we just grab all paragraphs
            paragraphs = page_soup.find_all("p")
            
            if not title_tag or len(paragraphs) < 3:
                continue
                
            title = title_tag.get_text().strip()
            
            text_chunks = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
            text = '\\n\\n'.join(text_chunks)
            
            if len(text) > 150:
                item = {
                    "url": url,
                    "title": title,
                    "text": text,
                    "source": "Dainik Bhaskar"
                }
                self.save_item(item)
                items_scraped += 1
                
                if items_scraped % 10 == 0:
                    self.logger.info(f"Scraped {items_scraped} documents")
                    
            if items_scraped >= max_items:
                break

if __name__ == "__main__":
    scraper = BhaskarScraper(output_dir="hindi/data/collection/raw/scraped")
    scraper.scrape(max_items=5) # Small test run
