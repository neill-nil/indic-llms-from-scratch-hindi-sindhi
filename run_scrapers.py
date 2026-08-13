import sys
import threading

from hindi.data.collection.scraping.gadyakosh_scraper import GadyakoshScraper
from hindi.data.collection.scraping.vikaspedia_scraper import VikaspediaScraper
from hindi.data.collection.scraping.bhaskar_scraper import BhaskarScraper
from sindhi.data.collection.scraping.sindhiana_scraper import SindhianaScraper
from sindhi.data.collection.scraping.awamiawaz_scraper import AwamiAwazScraper
from sindhi.data.collection.scraping.salamat_scraper import SalamatScraper

def run_scraper(scraper, max_items):
    try:
        scraper.scrape(max_items=max_items)
    except Exception as e:
        print(f"Error running {scraper.source_name}: {e}")

def main():
    max_items = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"Starting scraping sample run: {max_items} items per source...")
    
    scrapers = [
        GadyakoshScraper("hindi/data/collection/raw/scraped"),
        VikaspediaScraper("hindi/data/collection/raw/scraped"),
        BhaskarScraper("hindi/data/collection/raw/scraped"),
        SindhianaScraper("sindhi/data/collection/raw/scraped"),
        AwamiAwazScraper("sindhi/data/collection/raw/scraped"),
        SalamatScraper("sindhi/data/collection/raw/scraped")
    ]
    
    threads = []
    for s in scrapers:
        t = threading.Thread(target=run_scraper, args=(s, max_items))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print("Sampling complete! Check JSONL files in raw/scraped directories.")

if __name__ == "__main__":
    main()
