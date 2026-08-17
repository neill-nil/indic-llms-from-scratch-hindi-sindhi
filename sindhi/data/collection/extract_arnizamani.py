import os
import json
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent / "raw"
CLONE_DIR = RAW_DIR / "Sindhi-texts-big-dataset"

if not CLONE_DIR.exists():
    print("Clone directory not found. Please clone the dataset first.")
    exit(1)

SOURCES = [
    {
        "id": "arnizamani_books",
        "patterns": ["books.sindhsalamat.com"],
        "is_jsonl": False
    },
    {
        "id": "arnizamani_literature",
        "patterns": ["sindhiadabiboard.org"],
        "is_jsonl": False
    },
    {
        "id": "arnizamani_newspapers",
        "patterns": ["newspapers"],
        "is_jsonl": False
    },
    {
        "id": "arnizamani_dictionary",
        "patterns": ["jamesindhilughaat"],
        "is_jsonl": False
    },
    {
        "id": "arnizamani_sindhiana",
        "patterns": ["sindhiana.org"],
        "is_jsonl": False
    },
    {
        "id": "arnizamani_blogs",
        "patterns": ["blogs/bhurgri.com.txt"],
        "is_jsonl": False
    },
    {
        "id": "arnizamani_fineweb2",
        "patterns": ["fineweb2/fineweb2_snd.jsonl"],
        "is_jsonl": True,
        "text_key": "text"
    }
]

for source in SOURCES:
    out_file = RAW_DIR / f"{source['id']}.txt"
    if out_file.exists():
        print(f"Skipping {source['id']}, already exists.")
        continue

    print(f"Processing {source['id']}...")
    bytes_written = 0
    doc_count = 0
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        for pattern in source["patterns"]:
            search_path = CLONE_DIR / pattern
            
            if search_path.is_file():
                files_to_process = [search_path]
            elif search_path.is_dir():
                files_to_process = []
                for root, _, files in os.walk(search_path):
                    for file in files:
                        if file.endswith(".txt") or file.endswith(".jsonl"):
                            files_to_process.append(Path(root) / file)
            else:
                files_to_process = []
                
            for file_path in tqdm(files_to_process, desc=f"Reading {pattern}"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        if source["is_jsonl"]:
                            for line in f_in:
                                if not line.strip(): continue
                                try:
                                    data = json.loads(line)
                                    text = data.get(source.get("text_key", "text"), "").strip()
                                    if text:
                                        f_out.write(text + "\n\n")
                                        bytes_written += len((text + "\n\n").encode("utf-8"))
                                        doc_count += 1
                                except json.JSONDecodeError:
                                    pass
                        else:
                            text = f_in.read().strip()
                            if text:
                                f_out.write(text + "\n\n")
                                bytes_written += len((text + "\n\n").encode("utf-8"))
                                doc_count += 1
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    
    print(f"Finished {source['id']}: {doc_count} docs, {bytes_written / 1024 / 1024:.2f} MB")
