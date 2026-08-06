#!/bin/bash
# Start a vLLM server that can actually emit tool calls.
#
# Follows the 2025 module's setup (2025/Agents/vllm/start_vllm.sh), which is
# known to work on these VMs.
#
# THE ONE THING THAT MATTERS: --enable-auto-tool-choice --tool-call-parser.
# Without them vLLM starts fine, answers fine, and never emits a tool call. The
# agent loop then falls through to `no_tool_call` on every task and the
# afternoon is gone. Run `uv run python llm.py check` from your laptop after
# starting: it fails loudly if tool calls are missing.
#
# Run this ON THE GPU VM, inside tmux:
#
#   export HF_TOKEN='hf_...'        # needed to download weights
#   bash serve.sh
#
#   MODEL=Qwen/Qwen3.5-4B bash serve.sh   # smaller + much faster to start
set -euo pipefail

# ---------------------------------------------------------------------------
# NEWER THAN THE LIST WE WERE GIVEN — Qwen3.6 (April 2026, Apache 2.0) is the
# same architecture family as 3.5 and takes the same flags, but is explicitly
# tuned for agentic and tool-heavy work: 73.4 SWE-bench Verified, 52.6
# QwenClawBench, 37.0 MCPMark for the A3B. Needs **vLLM 0.19.0+**.
#
#   Qwen/Qwen3.6-35B-A3B-FP8   *** CLASS DEFAULT ***  35B total, 3B ACTIVE
#   Qwen/Qwen3.6-27B-FP8       dense, stronger still (77.2 SWE-bench), ~9x slower
#
# Fallback if 3.6 weights are not cached on the image, or vLLM is older —
# identical interface, one env var:
#
#   Qwen/Qwen3.5-35B-A3B-FP8   35B total, 3B ACTIVE.
#                              Decode on an L40S is memory-bandwidth bound, so
#                              speed tracks *active* parameters -- this answers
#                              roughly an order of magnitude faster than a dense
#                              32B while staying competitive. In an afternoon of
#                              iterating against an eval, throughput is the
#                              binding constraint on how many times you get to
#                              be wrong. Also: this is the architecture you
#                              implemented by hand on Monday.
#   Qwen/Qwen3.5-4B            fastest; use it while debugging your loop
#   Qwen/Qwen3.5-27B-FP8       dense, stronger, slow
#   Qwen/Qwen3-32B-FP8         what the 2025 class ran; proven on this hardware
#   google/gemma-4-12B-it      needs a different --tool-call-parser; untested here
#   google/gemma-4-26B-A4B-it  same caveat
#
# Stay on the Qwen family unless you enjoy debugging tool-call parsers. `hermes`
# is proven for Qwen; Gemma's chat template is different and `hermes` will not
# parse it.
#
# When you post a result, name the model. A score from a 27B dense model
# is not comparable to one from a 3B-active MoE -- which is the morning's lesson
# arriving in your own competition.
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-8000}"
# Bind to loopback. These VMs have public IPs and vLLM has no authentication,
# so 0.0.0.0 would put an open inference endpoint on the internet. To use it
# from your laptop, forward the port instead:
#     ssh -L 8000:localhost:8000 student@<vm-ip>
# then http://localhost:8000/v1 works locally.  HOST=0.0.0.0 to override.
HOST="${HOST:-127.0.0.1}"
MAX_LEN="${MAX_LEN:-16384}"
# Hybrid models (Qwen3.5/3.6 use Gated DeltaNet alongside attention) allocate an
# SSM state cache per sequence, separate from the KV cache. vLLM's default of 256
# concurrent sequences does not fit on a 46 GB card and it refuses to start:
#   "max_num_seqs (256) exceeds available Mamba cache blocks (173)"
# A classroom needs nothing like 256 — the evaluator runs 8 at a time.
MAX_SEQS="${MAX_SEQS:-64}"

# Tool-call parser must match the model family's chat template. Getting this
# wrong is the silent killer: the server starts, answers fluently, and never
# emits a tool call.
#
#   Qwen3.5 / Qwen3.6 -> qwen3_coder   (+ --reasoning-parser qwen3)
#   Qwen3 / Qwen2.5   -> hermes        (what the 2025 class used)
#   Llama 3.x         -> llama3_json
#   Mistral           -> mistral
#   Gemma             -> neither of the above; untested here
case "$MODEL" in
*Qwen3.5* | *Qwen3.6*) DEFAULT_PARSER="qwen3_coder"; DEFAULT_REASONING="qwen3" ;;
*Qwen3* | *Qwen2.5*) DEFAULT_PARSER="hermes"; DEFAULT_REASONING="" ;;
*[Ll]lama*) DEFAULT_PARSER="llama3_json"; DEFAULT_REASONING="" ;;
*istral*) DEFAULT_PARSER="mistral"; DEFAULT_REASONING="" ;;
*) DEFAULT_PARSER="hermes"; DEFAULT_REASONING="" ;;
esac
PARSER="${PARSER:-$DEFAULT_PARSER}"
REASONING="${REASONING-$DEFAULT_REASONING}"

if [ -z "${HF_TOKEN:-}" ]; then
	echo "warning: HF_TOKEN is not set; gated weights will fail to download." >&2
	echo "         export HF_TOKEN='hf_...'" >&2
fi

# Belt and braces: use the PyTorch-native sampler rather than FlashInfer's
# JIT-compiled one. Costs a little throughput, removes a compile step that
# needs ninja and nvcc present and correct on every student VM.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "model  : $MODEL"
echo "parser : $PARSER"
echo "listen : $HOST:$PORT"
echo
echo "Once the server reports startup is complete, run this from your laptop:"
echo "  ARMLLM_BASE_URL=http://<this-vm-public-ip>:$PORT/v1 uv run python llm.py check"
echo

ARGS=(
	--host "$HOST"
	--port "$PORT"
	--max-model-len "$MAX_LEN"
	--max-num-seqs "$MAX_SEQS"
	--gpu-memory-utilization "${GPU_FRAC:-0.92}"
	--enable-auto-tool-choice
	--tool-call-parser "$PARSER"
)
[ -n "${HF_TOKEN:-}" ] && ARGS+=(--hf-token "$HF_TOKEN")
[ -n "$REASONING" ] && ARGS+=(--reasoning-parser "$REASONING")
# Escape hatch for "Model architectures [...] failed to be inspected": vLLM can
# fall back to the Transformers implementation for architectures it has no
# native kernel for. Slower, but it runs.  MODEL_IMPL=transformers bash serve.sh
[ -n "${MODEL_IMPL:-}" ] && ARGS+=(--model-impl "$MODEL_IMPL")
# KV cache, not weights, is the binding constraint on a 48 GB card once the
# model is ~35 GB. fp8 KV roughly doubles how many concurrent agent runs fit.
[ -n "${KV_FP8:-}" ] && ARGS+=(--kv-cache-dtype fp8)
# Skip CUDA graph capture. Slower per token, but it removes the memory spike at
# startup — the difference between booting and not on a 46 GB card holding 35 GB
# of weights.  EAGER=1 bash serve.sh
[ -n "${EAGER:-}" ] && ARGS+=(--enforce-eager)

exec vllm serve "$MODEL" "${ARGS[@]}"

# If the model's thinking text confuses the tool parser -- symptoms are
# intermittent tool calls, or arguments with stray prose in them -- add:
#   --reasoning-parser qwen3
#
# The 2025 script also pinned `transformers==4.53.3` and upgraded numba. That
# was a workaround for a specific breakage a year ago; try without it first and
# only reach for a pin if vLLM actually complains.
