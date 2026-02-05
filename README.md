# vLLM Quantization Bake-off (RTX 4070)

Single-GPU, single-machine project to understand inference trade-offs on constrained VRAM:
**FP16 vs BitsAndBytes INT8 vs BitsAndBytes INT4** using **vLLM** as the serving engine.

You will:
1) Launch an OpenAI-compatible vLLM server for each variant.
2) Run an async benchmark against it (TTFT, latency p50/p95, tokens/sec).
3) Sample peak VRAM via `nvidia-smi`.
4) Generate a markdown report.

> Target: RTX 4070-class GPU (often 12GB).

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Terminal A (server)
./scripts/serve.sh fp16

# Terminal B (benchmark)
python bench/bench.py --variant fp16 --num-prompts 20 --concurrency 4

# Repeat for bnb8 and bnb4:
./scripts/serve.sh bnb8
python bench/bench.py --variant bnb8 --num-prompts 20 --concurrency 4

./scripts/serve.sh bnb4
python bench/bench.py --variant bnb4 --num-prompts 20 --concurrency 4

# Optional deterministic sanity check:
python bench/quality_check.py --variant fp16
python bench/quality_check.py --variant bnb8
python bench/quality_check.py --variant bnb4

# Build report:
python scripts/report.py
```

Outputs:
- `results/<variant>_bench.json`
- `results/<variant>_quality.json`
- `results/report.md`

---

## Configuration

Edit `configs/model.yaml`:
- `model`: default `Qwen/Qwen2.5-3B-Instruct` (fits FP16 on 12GB)
- `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`

---

## Repo layout

- `scripts/serve.sh` — launch vLLM OpenAI server for a variant (fp16, bnb8, bnb4)
- `bench/bench.py` — async streaming benchmark client + metrics
- `bench/prompts.jsonl` — prompt set for benchmarking (editable)
- `bench/quality_check.py` — deterministic quality check
- `scripts/report.py` — builds `results/report.md` from JSON
- `scripts/nvidia_smi_sample.py` — standalone VRAM sampler
