import json
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def main():
    variants = ["fp16", "bnb8", "bnb4"]
    rows = []
    for v in variants:
        b = load_json(RESULTS_DIR / f"{v}_bench.json")
        q = load_json(RESULTS_DIR / f"{v}_quality.json")
        if not b:
            continue
        m = b["metrics"]
        rows.append({
            "variant": v,
            "model": b["model"],
            "peak_vram_mib": m.get("peak_vram_mib"),
            "ttft_p50": m.get("ttft_s_p50"),
            "ttft_p95": m.get("ttft_s_p95"),
            "lat_p50": m.get("latency_s_p50"),
            "lat_p95": m.get("latency_s_p95"),
            "tok_s_est": m.get("tok_per_s_est"),
            "quality": q.get("avg_score") if q else None,
        })

    out = []
    out.append("# vLLM Quantization Bake-off Report\n\n")
    out.append("This report summarizes **FP16 vs BitsAndBytes INT8 vs BitsAndBytes INT4** runs.\n\n")
    out.append("| Variant | Model | Peak VRAM (MiB) | TTFT p50 (s) | TTFT p95 (s) | Lat p50 (s) | Lat p95 (s) | Tok/s est | Quality (0..1) |\n")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        out.append(f"| {r['variant']} | {r['model']} | {fmt(r['peak_vram_mib'],0)} | {fmt(r['ttft_p50'])} | {fmt(r['ttft_p95'])} | {fmt(r['lat_p50'])} | {fmt(r['lat_p95'])} | {fmt(r['tok_s_est'])} | {fmt(r['quality'])} |\n")

    out.append("\n## Interpretation\n")
    out.append("- **Peak VRAM**: maximum memory used during the run.\n")
    out.append("- **TTFT**: time-to-first-token.\n")
    out.append("- **Tok/s est**: rough throughput estimate from output length.\n")
    out.append("- **Quality**: tiny deterministic sanity check.\n")

    out_path = RESULTS_DIR / "report.md"
    out_path.write_text("".join(out))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
