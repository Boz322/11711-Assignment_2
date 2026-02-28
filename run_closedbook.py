import json
import argparse
from pathlib import Path
from reader import load_model

QUERIES_FILE = "data/leaderboard_queries.json"
OUTPUT_FILE = "system_outputs/leaderboard_closedbook.json"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

def answer_closedbook(pipe, query):
    messages = [{"role": "system", "content": 
         "Answer the question concisely. One sentence or short phrase only."},
        {"role": "user", "content": f"Question: {query}"},
    ]
    out = pipe(messages)[0]["generated_text"]
    return out[-1]["content"].strip().split("\n")[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    queries = json.load(open(QUERIES_FILE))
    print(f"Loaded {len(queries)} queries")

    pipe = load_model(args.model)

    outputs = {}
    for item in queries:
        qid, q = item["id"], item["question"]
        a = answer_closedbook(pipe, q)
        outputs[qid] = a
        print(f"[{qid}] {q}\n -> {a}\n")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(outputs, open(OUTPUT_FILE, "w"), indent=2)
    print(f"Saved to {OUTPUT_FILE}")
    print("\nAdd 'andrewid' key before submitting to leaderboard!")
