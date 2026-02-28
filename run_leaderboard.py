import json
import argparse
from pathlib import Path
from retriever import load_index, retrieve
from reader import load_model, answer

ANDREW_ID = "bzhang3"
QUERIES_FILE = "data/leaderboard_queries.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["bm25","dense","hybrid"])
    args = parser.parse_args()
    output_file = f"system_outputs/leaderboard_{args.mode}.json"

    queries = json.load(open(QUERIES_FILE))
    print(f"Loaded {len(queries)} queries\n")

    # retrieval index + LLM
    bm25, emb_model, index, chunks = load_index()
    pipe = load_model()

    # run RAG
    output = {"andrewid": ANDREW_ID}
    for item in queries:
        qid, q = item["id"], item["question"]
        a = answer(pipe, q, bm25, emb_model, index, chunks, mode=args.mode)
        output[qid] =a
        print(f"[{qid}] {q}\n -> {a}\n")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(output_file, "w"), indent=2)
    print(f"Saved → {output_file}")