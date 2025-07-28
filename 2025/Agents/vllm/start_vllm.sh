pip install transformers==4.53.3
pip install --upgrade numba
vllm serve Qwen/Qwen3-32B-FP8 --hf-token $HF_TOKEN --host 0.0.0.0 --port 8000 --enable-auto-tool-choice --tool-call-parser hermes
