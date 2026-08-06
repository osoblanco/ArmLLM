# Setup — ArmLLM 2026, Day 4

Same shape as the 2025 agents module: **the model server runs on your GPU VM,
the code runs on your laptop.** If you get past `llm.py check` you are ready.

## 1. Start the server on your VM

```bash
ssh student@<your-vm-ip>
tmux new -s vllm                  # so it survives your ssh session
bash serve.sh
```

Your VM has an **NVIDIA L40S (48 GB)** and it is yours alone — you are not
sharing a rate limit with the room, so run as many evaluations as you like.

First start downloads weights and takes a few minutes. Wait for
`Application startup complete`, then detach with `Ctrl-b d`.

> `serve.sh` passes `--enable-auto-tool-choice --tool-call-parser hermes`.
> Those flags are the whole ballgame: without them vLLM starts, answers
> questions fluently, and never emits a single tool call — so your agent does
> nothing and the error message tells you nothing.

Prefer a faster start? `MODEL=Qwen/Qwen3-8B bash serve.sh`.

## 1b. Working from your laptop instead

Forward the port rather than exposing it — these VMs have public IPs and vLLM
has no authentication:

```bash
ssh -L 8000:localhost:8000 student@<your-vm-ip>
```

Leave that open, and `http://localhost:8000/v1` on your laptop reaches the
server. If you get `channel 3: open failed: connect failed`, forwarding is
disabled on the VM:

```bash
sudo sed -i 's/^AllowTcpForwarding no$/AllowTcpForwarding yes/' /etc/ssh/sshd_config
sudo systemctl restart ssh.service
```

## 2. Set up locally

On your laptop:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# add uv to PATH in ~/.bashrc or ~/.zshrc, then re-source it
git clone https://github.com/osoblanco/ArmLLM
cd ArmLLM/2026/agents
```

Everything runs through `uv run`. There is no environment to activate and no
`pip install` step — the first `uv run` builds it from `pyproject.toml`, later
ones are instant.

## 3. Point the code at your server

```bash
cp .env.example .env
```

Edit `.env`:

```
ARMLLM_PROFILE=vllm
ARMLLM_BASE_URL=http://<your-vm-public-ip>:8000/v1
ARMLLM_MODEL=Qwen/Qwen3-32B-FP8
ARMLLM_API_KEY=EMPTY
```

## 4. Check — do not skip this

```bash
uv run python llm.py check
```

Expected:

```
config  : profile=vllm model=Qwen/Qwen3-32B-FP8 base_url=http://.../v1
chat    : 'ok'   (0.4s)
tools   : search({"query": "zoning certificate fee Dvin"})   (0.9s)
usage   : {...}

OK — endpoint answers and speaks tool-calls.
```

**If `tools` says NO TOOL CALL RETURNED, stop and fix it before anything else.**
Every exercise today depends on it. Usual causes: the server was not started
through `serve.sh`, or the parser does not match the model family.

## Working without a GPU (online track)

```bash
ARMLLM_PROFILE=mock uv run python agent_solution.py "..."
```

runs the entire loop with no network, using a scripted policy. Good for reading
the code and for every evaluation exercise; useless for measuring a real model.
Or point `ARMLLM_BASE_URL` / `ARMLLM_MODEL` / `ARMLLM_API_KEY` at any
OpenAI-compatible hosted API.

## Never commit your key

`.env` is gitignored. If you fork this repo, keep it that way.

## Troubleshooting

| symptom | cause |
|---|---|
| `Connection refused` | server not up, or it is bound to localhost — `serve.sh` uses `--host 0.0.0.0` for a reason |
| `NO TOOL CALL RETURNED` | missing `--enable-auto-tool-choice`, or wrong `--tool-call-parser` for the model family |
| CUDA out of memory | `MAX_LEN=8192 bash serve.sh`, or use `MODEL=Qwen/Qwen3-8B` |
| weights fail to download | only gated models need `HF_TOKEN`; Qwen does not |
| agent always stops with `max_steps` | it is never calling `finish` — read the system prompt |
| tool arguments contain stray prose | reasoning text leaking into the parser; add `--reasoning-parser qwen3` |
| server dies when you close the terminal | you forgot `tmux` |
| `max_num_seqs (256) exceeds available Mamba cache blocks` | hybrid models allocate SSM state per sequence; `serve.sh` defaults `--max-num-seqs 64`, lower it with `MAX_SEQS=32` |
| `Model architectures [...] failed to be inspected` | missing `python3.12-dev`; Triton cannot compile. `bootstrap.sh` installs it |
| `Engine core initialization failed` | missing `ninja`; FlashInfer cannot compile its sampler. `bootstrap.sh` installs it |
| everything is slow (>5s per step) | reasoning tokens or eager mode — run `uv run python bench.py` |

## How fast should it be?

Measured on one L40S with `Qwen3.6-35B-A3B-FP8`, thinking off, CUDA graphs on:

| | |
|---|---|
| decode | ~100 tok/s |
| one tool call | ~1.5 s |
| one agent task (4 steps) | ~2.6 s |
| 40-task eval | well under a minute at concurrency 8 |

If you are far off that, `uv run python bench.py` tells you which layer is
responsible. Two settings account for a 12x difference between the worst and
best configuration of the *same model*: reasoning tokens, and CUDA graphs.
