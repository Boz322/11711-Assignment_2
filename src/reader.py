import json, argparse
from pathlib import Path
from retriever import load_index, retrieve

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TOP_K =5

def build_prompt(query, results):
    context = "\n\n".join(f"[{i+1}] {c['title']}: {c['text'][:600]}"
                          for i, (c, _) in enumerate(results))
    return (f"Answer concisely using only the context below.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")

def load_model(model_name=MODEL_NAME):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, do_sample=False)

def answer(pipe, query, bm25, emb_model, index, chunks, mode="hybrid"):
    results = retrieve(query, bm25, emb_model, index, chunks, mode=mode, k=TOP_K)
    context = "\n\n".join(f"[{i+1}] {c['title']}: {c['text'][:600]}"
                          for i, (c, _) in enumerate(results))
    messages = [{"role": "system", "content": "Answer concisely using only the context. One sentence or short phrase only."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},]
    out = pipe(messages)[0]["generated_text"]
    return out[-1]["content"].strip().split("\n")[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mode", default="hybrid", choices=["bm25","dense","hybrid"])
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    bm25, emb_model, index, chunks = load_index()
    pipe = load_model(args.model)

    if args.query:
        print(f"A: {answer(pipe, args.query, bm25, emb_model, index, chunks, args.mode)}")

    elif args.input:
        questions = [l.strip() for l in Path(args.input).read_text().splitlines() if l.strip()]
        outputs = {}
        for i, q in enumerate(questions, 1):
            a = answer(pipe, q, bm25, emb_model, index, chunks, args.mode)
            outputs[str(i)] = a
            print(f"[{i}/{len(questions)}] {q}\n  ->{a}")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            json.dump(outputs, open(args.output, "w"), indent=2)
            print(f"Saved to {args.output}")