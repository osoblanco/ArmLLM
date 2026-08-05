"""Student-controlled vLLM startup settings. The evaluator fixes the model."""

HOST = "127.0.0.1"
PORT = 8000
STARTUP_TIMEOUT_SECONDS = 600

VLLM_ARGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "40960",
    "--gpu-memory-utilization", "0.90",
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
]
