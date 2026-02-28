import json, pickle, argparse
import numpy as np
from pathlib import Path

INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = 5

def load_chunks():
    chunks = []
    with open("data/processed/chunks.jsonl") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

# BM25 
def build_bm25(chunks):
    import bm25s
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([c["text"] for c in chunks]))
    return retriever

def search_bm25(bm25, query, chunks, k=TOP_K):
    import bm25s
    results, scores = bm25.retrieve(bm25s.tokenize([query]), k=min(k, len(chunks)))
    return [(chunks[i], float(s)) for i, s in zip(results[0], scores[0])]


# FAISS

def build_faiss(chunks):
    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode([c["text"] for c in chunks], batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return model, index

def search_faiss(model, index, query, chunks, k=TOP_K):
    emb = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(emb, k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0],scores[0])]


# RRF Fusion
def rrf(bm25_results, faiss_results, k=60):
    scores, chunk_map = {}, {}
    for rank, (chunk, _) in enumerate(bm25_results + faiss_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0)+1 /(k + rank +1)
        chunk_map[cid] = chunk
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [(chunk_map[cid], scores[cid]) for cid in ranked]


# Build +Load +Search
def build_index():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print("Building BM25：")
    bm25 = build_bm25(chunks)
    pickle.dump(bm25, open(INDEX_DIR /"bm25.pkl", "wb"))

    print("Building FAISS：")
    model, index = build_faiss(chunks)
    model.save(str(INDEX_DIR / "embedder"))
    import faiss; faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    pickle.dump(chunks, open(INDEX_DIR / "chunks.pkl", "wb"))
    print(f"Done! Indexes saved to {INDEX_DIR}/")

def load_index():
    import faiss
    from sentence_transformers import SentenceTransformer
    bm25   = pickle.load(open(INDEX_DIR / "bm25.pkl", "rb"))
    model  = SentenceTransformer(str(INDEX_DIR /"embedder"))
    index  = faiss.read_index(str(INDEX_DIR /"faiss.index"))
    chunks = pickle.load(open(INDEX_DIR /"chunks.pkl", "rb"))
    return bm25, model, index, chunks


def retrieve(query, bm25, model, index, chunks, mode="hybrid", k=TOP_K):
    if mode == "bm25":  return search_bm25(bm25, query, chunks, k)
    if mode == "dense": return search_faiss(model, index, query, chunks, k)
    return rrf(search_bm25(bm25, query, chunks, k*2),
               search_faiss(model, index, query, chunks, k*2))[:k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--mode", default="hybrid", choices=["bm25","dense","hybrid"])
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        bm25, model, index, chunks = load_index()
        results = retrieve(args.query, bm25, model, index, chunks, args.mode)
        print(f"\nQuery: {args.query}  [{args.mode}]\n" + "="*60)
        for i, (chunk, score) in enumerate(results):
            print(f"\n[{i+1}] {chunk['title']} (score={score:.4f})")
            print(chunk['text'][:300] + "......")