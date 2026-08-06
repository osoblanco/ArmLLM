"""Score an agent, honestly.

Four numbers, and the day is an argument about why you need all four:

  accuracy     what everyone reports
  95% CI       40 items is +/- ~15 points; a 3-point "improvement" is noise
  pass^k       every one of k runs correct. Reliability, not luck. This is the
               number that collapses -- 61% pass@1 and 25% pass^8 is the same
               agent measured two ways
  tokens       an agent that is 2 points better and 5x more expensive is worse

Plus a failure histogram, because "62%" tells you nothing about what to fix,
and `stale` vs `hallucinated` vs `max_steps` tells you exactly what to fix.

    uv run python evaluate.py                       # all 40 tasks, 1 run each
    uv run python evaluate.py --repeats 3            # adds pass^3
    uv run python evaluate.py --ids mine-01 --verbose # just the one you wrote
    uv run python evaluate.py --agent agent_solution # score the reference
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm import LLM
from verify import verify

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks.jsonl"


def load_tasks(path: Path, limit: int | None = None, kinds: list[str] | None = None,
               ids: list[str] | None = None) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if ids:
        want = set(ids)
        tasks = [t for t in tasks if t["id"] in want]
        missing = want - {t["id"] for t in tasks}
        if missing:
            have = [json.loads(l)["id"] for l in
                    path.read_text(encoding="utf-8").splitlines() if l]
            sys.exit(
                f"\n  No task with id {', '.join(sorted(missing))} in {path.name}.\n"
                f"  It has {len(have)} tasks, {have[0]}..{have[-1]}.\n\n"
                f"  Writing your own? Append one line to {path.name} first:\n"
                f'    {{"id": "mine-01", "question": "...", '
                f'"verify": {{"type": "numeric", "value": 19500, "tol": 0.5}}}}\n')
    if kinds:
        tasks = [t for t in tasks if t.get("kind") in kinds]
    return tasks[:limit] if limit else tasks


def bootstrap_ci(flags: list[bool], iters: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI on a proportion. Resampling, so it needs no normal approximation
    and behaves sanely at n=40 and at accuracies near 0 or 1."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(iters):
        means.append(sum(flags[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def run_one(agent_cls, task: dict, temperature: float) -> dict:
    llm = LLM(temperature=temperature)
    agent = agent_cls(llm=llm)
    t0 = time.time()
    try:
        res = agent.run(task["question"])
        answer, stopped = res.answer, res.stopped_because
        usage, steps = res.usage.as_dict(), len(res.steps)
        trajectory = res.as_dict()["trajectory"]
        error = None
    except Exception as exc:  # noqa: BLE001 - a crashed run is a result, not a stop
        answer, stopped, error = "", "exception", f"{type(exc).__name__}: {exc}"
        usage, steps, trajectory = llm.usage.as_dict(), 0, []

    verdict = verify(answer, task["verify"]) if error is None else {
        "correct": False, "reason": "exception",
    }
    # Do not let one failure hide inside another. An empty answer is not a wrong
    # answer, and an agent that ran out of steps did not hallucinate.
    if not verdict["correct"]:
        if not answer.strip():
            verdict["reason"] = "no_answer"
        elif stopped == "max_steps":
            verdict["reason"] = "gave_up"
    return {
        "task_id": task["id"],
        "kind": task["kind"],
        "question": task["question"],
        "answer": answer,
        "correct": verdict["correct"],
        "reason": verdict["reason"],
        "stopped_because": stopped,
        "steps": steps,
        "tokens": usage["total_tokens"],
        "seconds": round(time.time() - t0, 2),
        "error": error,
        "trajectory": trajectory,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="agent", help="module exposing an Agent class")
    p.add_argument("--tasks", type=Path, default=TASKS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", nargs="*", default=None,
                   help="score only these task ids, e.g. --ids mine-01")
    p.add_argument("--kinds", nargs="*", default=None,
                   help="single_hop multi_hop numeric unanswerable contradiction")
    p.add_argument("--repeats", type=int, default=1, help="runs per task; >1 gives pass^k")
    p.add_argument("--concurrency", type=int, default=8,
               help="8 saturates one L40S; past that it just queues")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--inject", action="store_true",
                   help="milestone 4: add the poisoned documents to the corpus")
    p.add_argument("--out", type=Path, default=HERE / "results" / "latest.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.inject:
        import tools

        n = tools.reset_corpus(with_injections=True).n
        print(f"corpus includes injected documents ({n} docs total)\n")

    agent_cls = getattr(importlib.import_module(args.agent), "Agent")
    tasks = load_tasks(args.tasks, args.limit, args.kinds, args.ids)
    jobs = [(t, r) for t in tasks for r in range(args.repeats)]
    print(f"{len(tasks)} tasks x {args.repeats} run(s) = {len(jobs)} rollouts, "
          f"concurrency {args.concurrency}\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        runs = list(pool.map(lambda j: run_one(agent_cls, j[0], args.temperature), jobs))
    elapsed = time.time() - t0

    if args.verbose:
        for r in runs:
            mark = "ok  " if r["correct"] else "FAIL"
            print(f"  {mark} {r['task_id']} [{r['kind']:14s}] {r['reason']:26s} "
                  f"{r['tokens']:6d} tok  {r['answer'][:60]!r}")
        print()

    # ---- aggregate ----------------------------------------------------
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_task[r["task_id"]].append(r)

    flat = [r["correct"] for r in runs]
    accuracy = sum(flat) / len(flat)
    lo, hi = bootstrap_ci(flat)
    tokens = [r["tokens"] for r in runs]

    # pass^k: every run of that task correct. Reliability, not best-of.
    pass_k = sum(all(x["correct"] for x in v) for v in by_task.values()) / len(by_task)

    per_kind = {}
    for kind in sorted({t["kind"] for t in tasks}):
        sub = [r["correct"] for r in runs if r["kind"] == kind]
        klo, khi = bootstrap_ci(sub)
        per_kind[kind] = {"n": len(sub), "accuracy": sum(sub) / len(sub),
                          "ci": [round(klo, 3), round(khi, 3)]}

    reasons = Counter(r["reason"] for r in runs if not r["correct"])
    stops = Counter(r["stopped_because"] for r in runs)

    # ---- report -------------------------------------------------------
    w = 62
    print("=" * w)
    print(f"  accuracy      {accuracy:6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    if args.repeats > 1:
        print(f"  pass^{args.repeats}        {pass_k:6.1%}   all {args.repeats} runs correct")
        print(f"                         gap to accuracy: {accuracy - pass_k:+.1%}")
    print(f"  tokens/task   {statistics.mean(tokens):6.0f}   median "
          f"{statistics.median(tokens):.0f}, total {sum(tokens):,}")
    print(f"  wall clock    {elapsed:6.1f}s")
    print("=" * w)

    print("\n  by task kind")
    for kind, s in per_kind.items():
        bar = "#" * round(s["accuracy"] * 20)
        print(f"    {kind:14s} {s['accuracy']:5.0%}  n={s['n']:<3d} "
              f"[{s['ci'][0]:.0%},{s['ci'][1]:.0%}] {bar}")

    if reasons:
        print("\n  why it failed")
        for reason, count in reasons.most_common():
            note = {
                "stale": "answered from a superseded document",
                "hallucinated": "invented a figure instead of declining",
                "abstained_on_answerable": "declined when the answer was there",
                "abstained_but_invented_number": "declined but still quoted a number",
                "no_number": "no figure in the answer at all",
                "exception": "the run crashed",
                "gave_up": "never stopped searching; budget exhausted",
                "no_answer": "the agent returned nothing at all",
            }.get(reason, "")
            print(f"    {reason:30s} {count:3d}  {note}")

    if set(stops) - {"finish"}:
        print("\n  how it stopped")
        for s, c in stops.most_common():
            print(f"    {s:30s} {c:3d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "agent": args.agent,
        "model": LLM().describe(),
        "n_tasks": len(tasks), "repeats": args.repeats,
        "accuracy": accuracy, "ci95": [lo, hi],
        f"pass^{args.repeats}": pass_k,
        "mean_tokens": statistics.mean(tokens), "total_tokens": sum(tokens),
        "seconds": elapsed,
        "per_kind": per_kind,
        "failure_reasons": dict(reasons),
        "stopped_because": dict(stops),
        "runs": [{k: v for k, v in r.items() if k != "trajectory"} for r in runs],
    }, indent=2), encoding="utf-8")

    traj = args.out.with_suffix(".trajectories.jsonl")
    with traj.open("w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps({"task_id": r["task_id"], "correct": r["correct"],
                                 "reason": r["reason"], "trajectory": r["trajectory"]},
                                ensure_ascii=False) + "\n")

    # One row for the shared board. Not a ranking: the point is that accuracy,
    # cost, abstention and the model name travel together, because no one of
    # them means anything on its own.
    if sum(tokens) == 0:
        print("\n" + "!" * w)
        print("  0 model calls. The agent never ran, so these labels mean nothing.")
        print("  Implement the TODOs in agent.py, or score the reference with:")
        print("      uv run python evaluate.py --agent agent_solution")
        print("!" * w)
    print("\n" + "=" * w)
    print("  POST THIS ROW ON THE BOARD")
    print("=" * w)
    print(f"  {'team':<14s}{'accuracy':>9s}{'tokens/task':>13s}{'declines':>9s}"
          f"{'pass^k':>8s}   model")
    abst = per_kind.get("unanswerable", {}).get("accuracy")
    print(f"  {'YOUR NAME':<14s}{accuracy:>8.1%}{statistics.mean(tokens):>13.0f}"
          f"{(f'{abst:.0%}' if abst is not None else '-'):>9s}"
          f"{pass_k:>8.1%}   {LLM().model}")
    print(f"\n  results      {args.out.relative_to(HERE)}")
    print(f"  trajectories {traj.relative_to(HERE)}   <- milestone 2 reads these\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
