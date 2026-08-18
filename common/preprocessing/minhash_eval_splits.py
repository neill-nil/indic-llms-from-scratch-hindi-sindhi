import argparse
import logging
import time
from pathlib import Path
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOC_SEPARATOR = "<|endoftext|>"
NUM_PERMUTATIONS = 128
JACCARD_THRESHOLD = 0.8

def get_ngrams(text, n=5):
    words = text.split()
    if len(words) < n:
        return [text]
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def process_file(input_path, output_path):
    logger.info(f"Starting MinHash LSH deduplication on {input_path}")
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERMUTATIONS)
    total_docs = 0
    duplicate_count = 0
    start = time.time()
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
         
        buffer = []
        for line in tqdm(f_in):
            line = line.strip()
            if line == DOC_SEPARATOR:
                doc_text = "\n".join(buffer).strip()
                if doc_text:
                    total_docs += 1
                    m = MinHash(num_perm=NUM_PERMUTATIONS)
                    for shingle in get_ngrams(doc_text):
                        m.update(shingle.encode('utf-8'))
                    result = lsh.query(m)
                    if len(result) > 0:
                        duplicate_count += 1
                    else:
                        doc_id = f"doc_{total_docs}"
                        lsh.insert(doc_id, m)
                        f_out.write(doc_text + f"\n{DOC_SEPARATOR}\n")
                buffer = []
            else:
                buffer.append(line)
                
    logger.info(f"Deduplication complete in {time.time()-start:.1f}s")
    logger.info(f"Total processed: {total_docs}")
    logger.info(f"Fuzzy duplicates removed: {duplicate_count}")
    logger.info(f"Surviving documents: {total_docs - duplicate_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["hindi", "sindhi"], required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    input_path = base_dir / args.language / "data" / "preprocessed" / f"{args.split}.txt"
    output_path = base_dir / args.language / "data" / "preprocessed" / f"{args.split}_minhash.txt"
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
        
    process_file(input_path, output_path)

if __name__ == "__main__":
    main()
