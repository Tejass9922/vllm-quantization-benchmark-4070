import argparse, time, subprocess, json
from pathlib import Path

def sample(duration_s: float, interval_s: float):
    peak = 0
    series = []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
            cur = max(vals) if vals else 0
            peak = max(peak, cur)
            series.append({"t": time.time(), "mem_used_mib": cur})
        except Exception:
            pass
        time.sleep(interval_s)
    return peak, series

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--interval-s", type=float, default=0.5)
    ap.add_argument("--out", default="results/vram_series.json")
    args = ap.parse_args()

    peak, series = sample(args.duration_s, args.interval_s)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({"peak_vram_mib": peak, "series": series}, indent=2))
    print(f"peak_vram_mib={peak} wrote {args.out}")
