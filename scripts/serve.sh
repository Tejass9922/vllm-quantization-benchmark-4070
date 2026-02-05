#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-fp16}"

python scripts/print_config.py

MODEL="$(python -c "import yaml;print(yaml.safe_load(open('configs/model.yaml'))['model'])")"
HOST="$(python -c "import yaml;print(yaml.safe_load(open('configs/model.yaml'))['host'])")"
PORT="$(python -c "import yaml;print(yaml.safe_load(open('configs/model.yaml'))['port'])")"
DTYPE="$(python -c "import yaml;print(yaml.safe_load(open('configs/model.yaml'))['dtype'])")"
MAXLEN="$(python -c "import yaml;print(yaml.safe_load(open('configs/model.yaml'))['max_model_len'])")"
GMU="$(python -c "import yaml;print(yaml.safe_load(open('configs/model.yaml'))['gpu_memory_utilization'])")"
MAXSEQS="$(python -c "import yaml;print(yaml.safe_load(open('configs/model.yaml'))['max_num_seqs'])")"

QUANT_ARGS=""
if [[ "$VARIANT" == "bnb8" ]]; then
  QUANT_ARGS="--quantization bitsandbytes --load-format bitsandbytes --dtype $DTYPE"
elif [[ "$VARIANT" == "bnb4" ]]; then
  QUANT_ARGS="--quantization bitsandbytes --load-format bitsandbytes --dtype $DTYPE"
elif [[ "$VARIANT" == "fp16" ]]; then
  QUANT_ARGS="--dtype $DTYPE"
else
  echo "Unknown variant: $VARIANT (use fp16|bnb8|bnb4)"
  exit 1
fi

echo ""
echo "Launching vLLM OpenAI server"
echo "  variant: $VARIANT"
echo "  model:   $MODEL"
echo "  host:    $HOST"
echo "  port:    $PORT"
echo ""

python -m vllm.entrypoints.openai.api_server   --model "$MODEL"   --host "$HOST"   --port "$PORT"   --max-model-len "$MAXLEN"   --gpu-memory-utilization "$GMU"   --max-num-seqs "$MAXSEQS"   $QUANT_ARGS
