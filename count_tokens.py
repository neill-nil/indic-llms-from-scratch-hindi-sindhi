import sentencepiece as spm
import multiprocessing as mp
import os
import sys

def count_tokens_in_chunk(args):
    sp_model_path, file_path, start, end = args
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line: break
            count += len(sp.encode_as_ids(line))
    return count

def get_total_tokens(sp_model_path, file_path, num_workers=8):
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_workers
    chunks = []
    
    with open(file_path, 'rb') as f:
        for i in range(num_workers):
            start = i * chunk_size
            if start != 0:
                f.seek(start)
                f.readline()
                start = f.tell()
            
            end = (i + 1) * chunk_size if i < num_workers - 1 else file_size
            chunks.append((sp_model_path, file_path, start, end))
            
    with mp.Pool(num_workers) as pool:
        counts = pool.map(count_tokens_in_chunk, chunks)
        
    return sum(counts)

if __name__ == "__main__":
    lang = sys.argv[1]
    model_path = f"{lang}/tokenizer/{lang}_unigram_32000.model"
    train_path = f"{lang}/data/preprocessed/train.txt"
    val_path = f"{lang}/data/preprocessed/val.txt"
    test_path = f"{lang}/data/preprocessed/test.txt"
    
    print(f"Counting {lang.upper()}...")
    train_tokens = get_total_tokens(model_path, train_path)
    print(f"Train Tokens: {train_tokens:,}")
    val_tokens = get_total_tokens(model_path, val_path)
    print(f"Val Tokens: {val_tokens:,}")
    test_tokens = get_total_tokens(model_path, test_path)
    print(f"Test Tokens: {test_tokens:,}")
    print(f"Total: {train_tokens + val_tokens + test_tokens:,}")
