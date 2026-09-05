#!/usr/bin/env bash
# Serve the hindsight rewriter (DeepSeek-R1-Distill-Qwen-32B) over an OpenAI-compatible API.
#
# DEVIATION (antidistill): Kim et al. use SGLang (`start_hint_server.sh`). We use vLLM's
# OpenAI-compatible server instead — sglang is not installed and would contend with vLLM's pins
# in the same environment. The generator only calls `/v1/completions`, which both implement
# identically; model, prompt and sampling parameters are unchanged.
#
# TENSOR PARALLEL AND TOPOLOGY (host-dependent — check yours before copying this):
# TP does ~2 all-reduces per layer per token, so it is sensitive to interconnect quality.
# On OUR development host P2P is broken and cross-socket collectives run at ~0.8 GB/s versus
# ~4.7 GB/s within a NUMA node, which made two TP=4 replicas beat one TP=8 server. On a healthy
# host with NVLink the opposite is usually true: prefer a single TP=8 server.
# Run `nvidia-smi topo -m`: if all pairs show NV# / NODE you are fine with large TP.
#
# Usage:
#   scripts/40_serve_teacher.sh              # TP=4 on GPUs 0-3, port 8001
#   GPUS=4,5,6,7 PORT=8002 scripts/40_serve_teacher.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-8001}"
TP=$(awk -F',' '{print NF}' <<<"$GPUS")

mkdir -p logs
LOG="$REPO/logs/teacher_server_${PORT}.log"

echo "model : $MODEL"
echo "gpus  : $GPUS (TP=$TP)"
echo "port  : $PORT"
echo "log   : ${LOG#$REPO/}"

# Host-specific interconnect settings (see scripts/check_host.sh). Not hardcoded:
# NCCL_P2P_DISABLE=1 is required on hosts with broken P2P and harmful on healthy ones.
if [[ -f "$REPO/.host_profile" ]]; then source "$REPO/.host_profile"; fi

CUDA_VISIBLE_DEVICES="$GPUS" VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  "$REPO/.venv-infer/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  > "$LOG" 2>&1 &

echo "server pid $! — waiting for readiness ..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "READY after ~$((i*10))s"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 300; echo
    exit 0
  fi
  if grep -qiE "error|Traceback|out of memory" "$LOG" 2>/dev/null; then
    echo "SERVER FAILED — last lines:"; tail -20 "$LOG"; exit 1
  fi
  sleep 10
done
echo "TIMEOUT waiting for server; last lines:"; tail -20 "$LOG"; exit 1
