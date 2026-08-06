#!/bin/bash
# One command. Run it on your GPU VM and everything works.
#
#   bash bootstrap.sh
#
# Checks the GPU, installs uv, python headers, ninja and vLLM, starts a model
# server with the right tool-call parser, waits for it, writes .env, and proves
# the model actually emits tool calls before saying READY.
#
# If a start fails it does not give up: it retries with safer memory settings,
# then with a small model, then with a different tool-call parser. Only when
# every one of those has failed does it stop and print the reason.
#
# Safe to re-run at any time.
#
#   MODEL=Qwen/Qwen3.5-4B bash bootstrap.sh   pick the model yourself
#   SKIP_SERVE=1          bash bootstrap.sh   set up the code only
#   HF_TOKEN=hf_...       bash bootstrap.sh   only if you switch to a gated model
#   MAX_SEQS=32           bash bootstrap.sh   fewer concurrent sequences (default 64)
#   EAGER=1               bash bootstrap.sh   no CUDA graphs; ~6x slower, less memory
#   ARMLLM_THINKING=1                         keep reasoning tokens; ~5x slower
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
SESSION="${SESSION:-vllm}"
WAIT_SECS="${WAIT_SECS:-2400}"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok() { printf '  \033[32mok\033[0m    %s\n' "$1"; }
info() { printf '        %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
die() {
	printf '\n  \033[31mstopped\033[0m %s\n\n' "$1" >&2
	exit 1
}

# ---------------------------------------------------------------- progress UI
TTY=0
[ -t 1 ] && TTY=1
case "${LANG:-}${LC_ALL:-}" in
*UTF-8* | *utf8*) SPIN=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏); FULL='█'; EMPTY='░' ;;
*) SPIN=('|' '/' '-' '\\'); FULL='#'; EMPTY='.' ;;
esac

_bar() {
	local pct=$1 w=${2:-24} i filled out=''
	filled=$((pct * w / 100))
	for ((i = 0; i < w; i++)); do
		if [ "$i" -lt "$filled" ]; then out="$out$FULL"; else out="$out$EMPTY"; fi
	done
	printf '%s' "$out"
}
_clock() { printf '%02d:%02d' $(($1 / 60)) $(($1 % 60)); }

LAST_PHASE=''
draw() {
	local frame=${SPIN[$(($1 % ${#SPIN[@]}))]} line
	if [ "$4" -ge 0 ]; then
		line=$(printf '  %s  %s  %-22s %s %3d%%' "$frame" "$(_clock "$2")" "$3" "$(_bar "$4")" "$4")
	else
		line=$(printf '  %s  %s  %-22s' "$frame" "$(_clock "$2")" "$3")
	fi
	if [ "$TTY" -eq 1 ]; then
		printf '\r\033[K%s' "$line"
	elif [ "$3" != "$LAST_PHASE" ]; then
		printf '        %s\n' "$3"
		LAST_PHASE="$3"
	fi
}
endline() { [ "$TTY" -eq 1 ] && printf '\r\033[K'; }

phase_from_log() {
	local log=$1 tail_txt pct
	[ -f "$log" ] || { echo "starting|-1"; return; }
	tail_txt=$(tail -c 4000 "$log" 2>/dev/null | tr '\r' '\n')
	if grep -qE '^INFO: +Application startup complete' <<<"$tail_txt"; then
		echo "ready|100"
	elif grep -qE 'Capturing CUDA graph|torch.compile|Compiling' <<<"$tail_txt"; then
		echo "optimising|-1"
	elif pct=$(grep -oE 'Loading safetensors checkpoint shards: *[0-9]+%' <<<"$tail_txt" | tail -1 | grep -oE '[0-9]+'); then
		echo "loading weights|${pct:--1}"
	elif pct=$(grep -E 'safetensors|\.bin' <<<"$tail_txt" | grep -oE '[0-9]+%' | tail -1 | tr -d '%'); then
		echo "downloading weights|${pct:--1}"
	elif grep -q 'Starting vLLM' <<<"$tail_txt"; then
		echo "starting engine|-1"
	else
		echo "starting|-1"
	fi
}

spin_run() {
	local label=$1; shift 2
	local t0 tick=0 pid rc
	t0=$(date +%s)
	("$@" >"$HERE/.bootstrap-step.log" 2>&1) &
	pid=$!
	while kill -0 "$pid" 2>/dev/null; do
		draw "$tick" "$(($(date +%s) - t0))" "$label" -1
		tick=$((tick + 1))
		sleep 0.15
	done
	wait "$pid"; rc=$?
	endline
	return $rc
}

log_reason() { # last real line of the server's traceback
	grep 'EngineCore pid\|ERROR' "$HERE/vllm.log" 2>/dev/null |
		grep -vE '\^+\s*$' | tail -1 | sed 's/.*\] //' | cut -c1-100
}

# --------------------------------------------------------------------- 1. host
bold "ArmLLM 2026 · Day 4 — setup"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
	ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
	warn "no GPU here — every exercise still works with ARMLLM_PROFILE=mock"
	SKIP_SERVE=1
fi

# ----------------------------------------------------------------------- 2. uv
if ! command -v uv >/dev/null 2>&1; then
	info "installing uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
	grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null ||
		echo 'export PATH="$HOME/.local/bin:$PATH"' >>"$HOME/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv install failed. Try: pip install uv"
ok "uv $(uv --version | awk '{print $2}')"

uv python find ">=3.10" >/dev/null 2>&1 || uv python install 3.12 >/dev/null 2>&1
info "building the environment..."
uv sync --quiet || die "uv sync failed. Run 'uv sync' here to see why."
ok "environment ready"

# ------------------------------------------------------- 3. build dependencies
# Both of tonight's failures were a missing build tool reported as something
# else entirely: no Python.h -> "Model architectures failed to be inspected";
# no ninja -> "Engine core initialization failed" from determine_available_memory.
if [ "${SKIP_SERVE:-}" != "1" ]; then
	pyver=$(python3 -c 'import sys;print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 3.12)
	need=""
	[ -f "/usr/include/python$pyver/Python.h" ] || need="$need python$pyver-dev"
	command -v ninja >/dev/null 2>&1 || need="$need ninja-build"
	command -v gcc >/dev/null 2>&1 || need="$need build-essential"
	if [ -n "$need" ]; then
		APT="sudo -n apt-get -o DPkg::Lock::Timeout=180 -qq"
		spin_run "apt:$need" -- bash -c "$APT update >/dev/null 2>&1; $APT install -y $need >/dev/null 2>&1" || true
	fi
	for t in gcc ninja; do
		command -v $t >/dev/null 2>&1 || warn "$t is still missing — vLLM may fail to start"
	done
	[ -f "/usr/include/python$pyver/Python.h" ] && ok "build tools ready" ||
		{ info "no python headers — using uv's own python, which ships them"
		  export UV_PYTHON_PREFERENCE=only-managed; }

	if ! command -v vllm >/dev/null 2>&1; then
		spin_run "installing vllm (~5 GB)" -- uv tool install "vllm${VLLM_VERSION:+==$VLLM_VERSION}" ||
			tail -5 "$HERE/.bootstrap-step.log" >&2
		export PATH="$HOME/.local/bin:$PATH"
	fi
	command -v vllm >/dev/null 2>&1 || die "could not install vllm.
        Everything still works offline with: ARMLLM_PROFILE=mock"
	vv=$(vllm --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
	ok "vllm ${vv:-unknown}"
	case "$MODEL:$vv" in
	*Qwen3.6*:0.1[0-8].*) warn "vLLM $vv predates Qwen3.6; using Qwen3.5 instead"
		MODEL="Qwen/Qwen3.5-35B-A3B-FP8" ;;
	esac
fi

# ------------------------------------------------------------------- 4. server
kill_server() {
	# tmux kill-session only ends the shell; the engine workers keep the GPU.
	tmux kill-session -t "$SESSION" 2>/dev/null || true
	pkill -f 'vllm|VLLM::' 2>/dev/null && sleep 4
	pkill -9 -f 'vllm|VLLM::' 2>/dev/null && sleep 2
	rm -f "$HERE/vllm.pid"
	return 0
}

start_server() { # $1 model, $2 extra env  -> 0 when it is serving
	local model=$1 extra=${2:-} t0 waited=0 tick=0 phase=starting pct=-1 alive
	kill_server
	: >"$HERE/vllm.log"
	local cmd="MODEL='$model' PORT='$PORT' HF_TOKEN='${HF_TOKEN:-}' $extra bash '$HERE/serve.sh'"
	if command -v tmux >/dev/null 2>&1; then
		tmux new-session -d -s "$SESSION" "$cmd 2>&1 | tee '$HERE/vllm.log'"
	else
		nohup bash -c "$cmd" >"$HERE/vllm.log" 2>&1 &
		echo $! >"$HERE/vllm.pid"
	fi
	t0=$(date +%s)
	until curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
		alive=1
		if [ -f "$HERE/vllm.pid" ]; then
			kill -0 "$(cat "$HERE/vllm.pid")" 2>/dev/null || alive=0
		elif command -v tmux >/dev/null 2>&1; then
			tmux has-session -t "$SESSION" 2>/dev/null || alive=0
		fi
		[ "$alive" -eq 0 ] && { endline; return 1; }
		[ "$waited" -ge "$WAIT_SECS" ] && { endline; return 1; }
		[ $((tick % 8)) -eq 0 ] && IFS='|' read -r phase pct <<<"$(phase_from_log "$HERE/vllm.log")"
		draw "$tick" "$waited" "$phase" "${pct:--1}"
		tick=$((tick + 1))
		sleep 0.25
		waited=$(($(date +%s) - t0))
	done
	endline
	ok "server answering after $(_clock "$waited")  [$model $extra]"
	SERVED_MODEL="$model"
	return 0
}

SERVED_MODEL=""
if [ "${SKIP_SERVE:-}" = "1" ]; then
	warn "skipping the model server"
elif curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
	ok "a server is already running on port $PORT"
	SERVED_MODEL="${ARMLLM_MODEL:-$MODEL}"
else
	# The ladder: ideal, then memory-safe, then a small model that always fits.
	if start_server "$MODEL" ""; then :
	else
		warn "that start failed: $(log_reason)"
		info "retrying with conservative memory settings..."
		if start_server "$MODEL" "EAGER=1 GPU_FRAC=0.85 MAX_LEN=8192 KV_FP8=1 MAX_SEQS=16"; then :
		else
			warn "still failing: $(log_reason)"
			info "falling back to $SMALL_MODEL, which always fits..."
			start_server "$SMALL_MODEL" "" ||
				die "could not start any model. Last error:
        $(log_reason)

        full log: $HERE/vllm.log
        Everything still works offline with: ARMLLM_PROFILE=mock"
		fi
	fi
fi

# ---------------------------------------------------------------------- 5. env
if [ "${SKIP_SERVE:-}" = "1" ]; then
	[ -f .env ] || cp .env.example .env
else
	cat >.env <<EOF
ARMLLM_PROFILE=vllm
ARMLLM_BASE_URL=http://localhost:$PORT/v1
ARMLLM_MODEL=$SERVED_MODEL
ARMLLM_API_KEY=EMPTY
EOF
	ok "wrote .env for $SERVED_MODEL"
fi

# ------------------------------------------------------------------- 6. verify
echo
bold "checking that the model speaks tool-calls"
echo
if ! uv run python llm.py check; then
	if [ "${SKIP_SERVE:-}" != "1" ]; then
		echo
		warn "no tool calls — retrying with the hermes parser"
		echo
		if start_server "$SERVED_MODEL" "PARSER=hermes" && uv run python llm.py check; then
			ok "hermes parser works for this model"
		else
			die "the model does not emit tool calls, so nothing downstream works.
        See the parser table in serve.sh, or use ARMLLM_PROFILE=mock."
		fi
	else
		die "mock backend check failed — that should not happen."
	fi
fi

echo
bold "READY"
echo
info "Next:  uv run python agent.py --test"
info "Then:  open README.md and start at milestone 1."
echo
