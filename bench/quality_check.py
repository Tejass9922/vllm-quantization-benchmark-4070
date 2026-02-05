import argparse, json, os, time
from pathlib import Path
import yaml
from openai import OpenAI

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

QA = [
    {"q": "What is quantization in LLM inference?", "must_include": ["precision", "int", "memory"]},
    {"q": "What does p95 latency mean?", "must_include": ["95", "percentile"]},
    {"q": "Why do people use batching in LLM serving?", "must_include": ["throughput", "tokens"]},
]

def load_cfg():
    return yaml.safe_load(open("configs/model.yaml"))

def score(resp: str, must_include):
    s = (resp or "").lower()
    hits = sum(1 for w in must_include if w in s)
    return hits / max(1, len(must_include))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fp16", "bnb8", "bnb4"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()

    cfg = load_cfg()
    model = cfg["model"]

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    rows = []
    for item in QA:
        t0 = time.time()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":item["q"]}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )
        text = r.choices[0].message.content or ""
        rows.append({"q": item["q"], "resp": text, "score": score(text, item["must_include"]), "latency_s": time.time() - t0})

    avg = sum(x["score"] for x in rows)/len(rows)
    out = {"variant": args.variant, "model": model, "avg_score": avg, "rows": rows, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path = RESULTS_DIR / f"{args.variant}_quality.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} (avg_score={avg:.2f})")
