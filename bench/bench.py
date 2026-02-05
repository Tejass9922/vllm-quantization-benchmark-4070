import argparse, asyncio, json, os, time, subprocess
from pathlib import Path
import yaml
import numpy as np
from openai import AsyncOpenAI

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_cfg():
    return yaml.safe_load(open("configs/model.yaml"))

def load_prompts(path="bench/prompts.jsonl", limit=None):
    prompts = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            prompts.append(json.loads(line)["prompt"])
            if limit and len(prompts) >= limit:
                break
    return prompts

async def sample_vram(duration_s: float, interval_s: float = 0.5) -> int:
    peak = 0
    t_end = time.time() + duration_s
    while time.time() < t_end:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
            if vals:
                peak = max(peak, max(vals))
        except Exception:
            pass
        await asyncio.sleep(interval_s)
    return peak

async def one_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float):
    start = time.time()
    ttft = None
    out_text = []
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    async for chunk in stream:
        now = time.time()
        if ttft is None:
            ttft = now - start
        delta = chunk.choices[0].delta
        if delta and delta.content:
            out_text.append(delta.content)
    total = time.time() - start
    text = "".join(out_text)
    chars = len(text)
    tok_est = max(1, int(chars / 4)) if chars else 0
    return (ttft or total), total, tok_est, chars

def pct(values, p):
    if not values:
        return None
    return float(np.percentile(np.array(values), p))

async def run_bench(args):
    cfg = load_cfg()
    model = cfg["model"]
    gen = cfg.get("gen", {})
    max_tokens = args.max_tokens if args.max_tokens is not None else int(gen.get("max_tokens", 256))
    temperature = args.temperature if args.temperature is not None else float(gen.get("temperature", 0.2))
    top_p = args.top_p if args.top_p is not None else float(gen.get("top_p", 0.95))
    prompts = load_prompts(limit=args.num_prompts)

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)

    # Warmup
    await one_request(client, model, prompts[0], max_tokens=min(32, max_tokens), temperature=0.0, top_p=1.0)

    sem = asyncio.Semaphore(args.concurrency)
    ttfts, totals, token_ests, char_counts = [], [], [], []

    async def bounded(prompt):
        async with sem:
            return await one_request(client, model, prompt, max_tokens, temperature, top_p)

    load_duration_est = max(10.0, args.num_prompts * 0.5)
    vram_task = asyncio.create_task(sample_vram(duration_s=load_duration_est + 5.0))

    t0 = time.time()
    tasks = [asyncio.create_task(bounded(p)) for p in prompts]
    for coro in asyncio.as_completed(tasks):
        ttft, total, tok_est, chars = await coro
        ttfts.append(ttft)
        totals.append(total)
        token_ests.append(tok_est)
        char_counts.append(chars)
    wall = time.time() - t0
    peak_vram = await vram_task

    total_tok_est = sum(token_ests)
    tok_per_s_est = total_tok_est / wall if wall > 0 else 0.0

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant": args.variant,
        "model": model,
        "base_url": args.base_url,
        "num_prompts": len(prompts),
        "concurrency": args.concurrency,
        "gen": {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
        "metrics": {
            "ttft_s_p50": pct(ttfts, 50),
            "ttft_s_p95": pct(ttfts, 95),
            "latency_s_p50": pct(totals, 50),
            "latency_s_p95": pct(totals, 95),
            "wall_s": wall,
            "tok_per_s_est": tok_per_s_est,
            "total_tok_est": total_tok_est,
            "total_chars": int(sum(char_counts)),
            "peak_vram_mib": int(peak_vram),
        },
    }

    out_path = RESULTS_DIR / f"{args.variant}_bench.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fp16", "bnb8", "bnb4"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    args = ap.parse_args()
    asyncio.run(run_bench(args))
