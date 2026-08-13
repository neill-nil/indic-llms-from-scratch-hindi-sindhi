import sys
from bs4 import BeautifulSoup
from common.scraping.base_scraper import BaseScraper

class VikaspediaScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir=output_dir, source_name="vikaspedia", delay=0.5)
        # Using the health section as an example for structured data
        self.start_url = "https://hi.vikaspedia.in/health"

    def scrape(self, max_items: int = 1000):
        self.logger.info(f"Starting Vikaspedia scraper for {max_items} items")
        items_scraped = 0
        
        # In a real comprehensive scraper, we would crawl all internal links.
        # For this demo, we'll just extract the main page texts and links.
        soup = self.fetch_page(self.start_url)
        if not soup:
            return
            
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/health/'):
                links.add(f"https://hi.vikaspedia.in{href}")
                
        self.logger.info(f"Found {len(links)} initial links to scrape")
        
        for url in list(links)[:max_items]:
            page_soup = self.fetch_page(url)
            if not page_soup:
                continue
                
            title_tag = page_soup.find("h1")
            content_div = page_soup.find(id="textcontent") or page_soup.find("article")
            
            if not title_tag or not content_div:
                continue
                
            title = title_tag.get_text().strip()
            
            # Clean up
            for tag in content_div.find_all(['script', 'style', 'div']):
                tag.decompose()
                
            text = content_div.get_text(separator='\\n').strip()
            
            if len(text) > 100:
                item = {
                    "url": url,
                    "title": title,
                    "text": text,
                    "source": "Vikaspedia"
                }
                self.save_item(item)
                items_scraped += 1
                
                if items_scraped % 10 == 0:
                    self.logger.info(f"Scraped {items_scraped} documents")
                    
            if items_scraped >= max_items:
                break

if __name__ == "__main__":
    scraper = VikaspediaScraper(output_dir="hindi/data/collection/raw/scraped")
    scraper.scrape(max_items=10) # Small test run
