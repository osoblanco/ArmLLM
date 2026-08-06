"""Measure the server, so you tune the thing that is actually slow.

    uv run python bench.py            latency, decode rate, one agent task
    uv run python bench.py --full     + how throughput scales with concurrency

Why this exists: "the agent feels slow" is not a measurement. It could be
prefill, decode, kernel-launch overhead, thinking tokens, or your own loop
doing eight steps where three would do. Those have different fixes, and three
of them are free.

Useful comparisons to run:

    ARMLLM_THINKING=1 uv run python bench.py    thinking on vs off
    (restart the server with/without EAGER=1)   CUDA graphs on vs off
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from llm import LLM

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
FILLER = ("The Dvin municipal archive holds records of departmental fee "
          "schedules, opening hours and service regulations. ")


def timed(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def one_call(llm: LLM, prompt: str, max_tokens: int) -> tuple[float, int]:
    llm.max_tokens = max_tokens
    before = llm.usage.completion_tokens
    dt, _ = timed(lambda: llm.chat([{"role": "user", "content": prompt}]))
    return dt, llm.usage.completion_tokens - before


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="also sweep concurrency")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    llm = LLM()
    print(f"\n{BOLD}{llm.describe()}{OFF}\n")

    print(f"{DIM}warming up...{OFF}")
    one_call(llm, "Reply with: ok", 8)

    # ---- latency vs prompt size (isolates prefill) -----------------------
    print(f"{BOLD}latency by prompt size{OFF}  {DIM}(64 output tokens each){OFF}")
    print(f"  {'prompt tok':>10}  {'seconds':>8}  {'out tok':>7}")
    sizes = [(0, "Reply with: ok."), (2000, None), (6000, None)]
    for approx, prompt in sizes:
        if prompt is None:
            prompt = (FILLER * (approx // len(FILLER) + 1))[:approx * 4]
            prompt += "\n\nReply with the single word: ok."
        times = []
        outs = 0
        for _ in range(args.repeats):
            dt, out = one_call(llm, prompt, 64)
            times.append(dt)
            outs = out
        print(f"  {approx:>10}  {statistics.median(times):>8.2f}  {outs:>7}")

    # ---- decode rate ----------------------------------------------------
    print(f"\n{BOLD}decode rate{OFF}")
    dt, out = one_call(llm, "Count slowly from 1 to 120, one number per line.", 400)
    rate = out / dt if dt else 0
    print(f"  {out} tokens in {dt:.2f}s  =  {BOLD}{rate:.0f} tok/s{OFF}")
    if rate < 40:
        print(f"  {DIM}low. eager mode (no CUDA graphs) is the usual cause on a"
              f" small-active MoE.{OFF}")

    # ---- a real agent task ----------------------------------------------
    print(f"\n{BOLD}one agent task, end to end{OFF}")
    from agent_solution import Agent

    a = Agent(llm=LLM())
    dt, res = timed(lambda: a.run(
        "What fee does Dvin currently charge for a commercial signage permit?"))
    steps = len(res.steps)
    print(f"  {steps} steps, {dt:.1f}s  ({dt / max(steps, 1):.1f}s per step)")
    print(f"  {res.usage.prompt_tokens} prompt + {res.usage.completion_tokens} "
          f"completion tokens over {res.usage.calls} calls")
    print(f"  {DIM}40 tasks x 4 steps at this rate: "
          f"{40 * dt / 60:.0f} min serial, "
          f"{40 * dt / 60 / 16:.1f} min at concurrency 16{OFF}")

    # ---- concurrency ----------------------------------------------------
    if args.full:
        print(f"\n{BOLD}throughput vs concurrency{OFF}  {DIM}(8 agent tasks){OFF}")
        print(f"  {'workers':>7}  {'seconds':>8}  {'tasks/min':>9}")
        q = "What is the fee for a zoning certificate in Dvin?"
        for workers in (1, 4, 8, 16):
            def run_one(_):
                return Agent(llm=LLM()).run(q)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run_one, range(8)))
            el = time.perf_counter() - t0
            print(f"  {workers:>7}  {el:>8.1f}  {8 / el * 60:>9.1f}")

    print(f"""
{BOLD}what to do with this{OFF}
  latency flat across prompt sizes   -> prefill is cheap, look at decode
  decode under ~40 tok/s             -> restart without EAGER=1 (CUDA graphs)
  many completion tokens per step    -> thinking is on; ARMLLM_THINKING=0
  many steps per task                -> your loop, not the server
  throughput flat past 4 workers     -> raise MAX_SEQS, or the GPU is saturated
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
