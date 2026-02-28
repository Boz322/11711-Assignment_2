import re
import json
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Fixed-size
def chunk_fixed(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks, start = [],0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# Sentence-aware
def chunk_sentence(text, target_chars=1200, overlap_sents=2):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >20]

    chunks, current, tail = [],[],[]
    for sent in sentences:
        if not current and tail:
            current = list(tail)
        current.append(sent)
        if sum(len(s) for s in current) >= target_chars:
            chunks.append(" ".join(current))
            tail = current[-overlap_sents:]
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


# Main
def load_docs(raw_dir):
    docs = []
    for path in sorted(raw_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        source, title, body_start = "", path.stem,0
        for i, line in enumerate(lines[:5]):
            if line.startswith("SOURCE: "):
                source = line[8:].strip(); body_start = i +1
            elif line.startswith("TITLE: "):
                title = line[7:].strip(); body_start = i +1
        body = "\n".join(lines[body_start:]).strip()
        if len(body) >200:
            docs.append({"source": source, "title": title, "body": body})
    return docs

if __name__ == "__main__":
    docs = load_docs(RAW_DIR)
    print(f"Loaded {len(docs)} documents")

    for strategy, fn in [("fixed", chunk_fixed), ("sentence",chunk_sentence)]:
        chunks, cid = [],0
        for doc in docs:
            for text in fn(doc["body"]):
                if len(text) >100:
                    chunks.append({
                        "id": f"{strategy}_{cid:06d}",
                        "source": doc["source"],
                        "title": doc["title"],
                        "text": text,
                        "strategy": strategy,
                    })
                    cid +=1

        out = PROCESSED_DIR /f"chunks_{strategy}.jsonl"
        with open(out, "w") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        avg = sum(len(c["text"]) for c in chunks) //len(chunks)
        print(f"[{strategy}] {len(chunks):,} chunks, avg {avg} chars →{out}")

    shutil.copy(PROCESSED_DIR / "chunks_sentence.jsonl", PROCESSED_DIR /"chunks.jsonl")
    print("\nDone! Default: chunks.jsonl (sentence strategy)")