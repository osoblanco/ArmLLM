"""Milestone 1 — make it work.

An agent is five things:

    policy  + tools + observation  + memory          + stopping rule
    (model)   (4)     (tool result)  (messages list)   (`finish`, or max_steps)

That is the whole idea. Everything else -- planners, multi-agent, memory
systems -- is an optimisation on top of this loop, and you should not reach for
any of it until you have measured a failure this loop cannot fix.

Fill in the six TODOs below. Run the sanity check as you go:

    uv run python agent.py --test          # no model needed, checks the wiring
    uv run python agent.py "What is the fee for a zoning certificate in Dvin?"

`agent_solution.py` is the reference. Try not to open it until you have made a
real attempt, or use it to check one piece at a time.

When every TODO is done and `--test` passes, go to README.md milestone 2.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from llm import LLM, Usage
from tools import TOOL_SCHEMAS, TOOLS

SYSTEM_PROMPT = """You are a municipal services assistant for the city of Dvin.

Answer the user's question using ONLY the Dvin document corpus, which you reach
through the tools provided. Search first, read the documents that look relevant,
then answer.

Rules:
- Give the final answer by calling the `finish` tool. Nothing else ends the task.
- If the documents do not contain the answer, say so explicitly in `finish`.
  Do not guess, and do not invent figures.
- Quote figures exactly as the documents give them."""

MAX_OBSERVATION_CHARS = 4000


@dataclass
class Step:
    n: int
    tool: str
    args: dict
    observation: str
    tokens_after: int


@dataclass
class Result:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_because: str = "finish"

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "stopped_because": self.stopped_because,
            "n_steps": len(self.steps),
            "usage": self.usage.as_dict(),
            "trajectory": [
                {"step": s.n, "tool": s.tool, "args": s.args,
                 "observation": s.observation[:600], "tokens_after": s.tokens_after}
                for s in self.steps
            ],
        }


class Agent:
    def __init__(self, llm: LLM | None = None, max_steps: int = 8,
                 system_prompt: str = SYSTEM_PROMPT) -> None:
        self.llm = llm or LLM()
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(self, question: str) -> Result:
        # ------------------------------------------------------------------
        # TODO 1 — memory.
        # The agent's entire memory is this list. Seed it with the system
        # prompt and the user's question, in the OpenAI message format:
        #     {"role": "system", "content": ...}
        #     {"role": "user",   "content": ...}
        messages: list[dict] = []  # <-- your code here
        # ------------------------------------------------------------------

        result = Result(question=question, answer="", usage=self.llm.usage)

        for n in range(1, self.max_steps + 1):
            # --------------------------------------------------------------
            # TODO 2 — the policy.
            # Ask the model what to do next. Pass the tool schemas so it can
            # answer with a tool call:  self.llm.chat(messages, tools=...)
            # Then append its reply to `messages` -- if you forget this, the
            # model re-decides from scratch every step and loops forever.
            msg: dict = {}  # <-- your code here
            # --------------------------------------------------------------

            calls = msg.get("tool_calls")
            if not calls:
                # No tool call: the model answered in prose instead of calling
                # `finish`. Accept the answer, but record that the stopping rule
                # did not fire. You will classify this in milestone 2.
                result.answer = msg.get("content") or ""
                result.stopped_because = "no_tool_call"
                return result

            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                # ----------------------------------------------------------
                # TODO 3 — the stopping rule.
                # If the model called `finish`, put its "answer" argument on
                # `result.answer`, append a Step, and RETURN. Nothing else
                # ends the task -- an agent whose loop cannot be stopped by
                # anything except max_steps is the 09:05 failure.
                # <-- your code here
                # ----------------------------------------------------------

                # ----------------------------------------------------------
                # TODO 4 — act.
                # Look the tool up in TOOLS and call it with **args.
                # A tool that raises must NOT crash the loop: turn the
                # exception into an observation string starting with "ERROR:"
                # and let the model read it and recover. Error recovery is a
                # capability; a traceback is not.
                observation = ""  # <-- your code here
                # ----------------------------------------------------------

                if not isinstance(observation, str):
                    observation = json.dumps(observation, ensure_ascii=False)

                # ----------------------------------------------------------
                # TODO 5 — observe, and pay for it.
                # Truncate the observation to MAX_OBSERVATION_CHARS, then
                # append it to `messages` as:
                #   {"role": "tool", "tool_call_id": call["id"],
                #    "name": name, "content": observation}
                # Every step re-sends this whole list, so an observation you
                # keep is one you pay for on every remaining step. That is why
                # cost grows faster than step count.
                # <-- your code here
                # ----------------------------------------------------------

                result.steps.append(
                    Step(n, name, args, observation, self.llm.usage.total_tokens)
                )

        # TODO 6 — the budget ran out. Record it honestly rather than
        # pretending there was an answer.
        result.stopped_because = "max_steps"
        result.answer = result.answer or "No answer: step budget exhausted."
        return result


# ---------------------------------------------------------------------------


def selftest() -> int:
    """Checks the wiring without needing a model. Run this first."""
    import os

    os.environ["ARMLLM_PROFILE"] = "mock"
    checks: list[tuple[str, bool, str]] = []

    res = Agent().run("What is the fee for a zoning certificate in Dvin?")
    checks.append(("agent returns a Result",
                   all(hasattr(res, a) for a in ("answer", "steps", "usage")), ""))
    checks.append(("it took at least one step", len(res.steps) > 0,
                   "TODO 2: are you calling the model and appending its reply?"))
    checks.append(("it produced an answer", bool(res.answer),
                   "TODO 3: does `finish` set result.answer and return?"))
    checks.append(("it stopped via finish", res.stopped_because == "finish",
                   f"stopped_because={res.stopped_because!r} -- TODO 3 or TODO 6"))
    checks.append(("it used a tool", any(s.tool != "finish" for s in res.steps),
                   "TODO 4: are you dispatching through TOOLS?"))
    checks.append(("observations were recorded", any(s.observation for s in res.steps),
                   "TODO 5: are you appending the tool result to messages?"))
    checks.append(("tokens were counted", res.usage.total_tokens > 0, ""))

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, passed, hint in checks:
        mark = "\033[32mok\033[0m  " if passed else "\033[31mFAIL\033[0m"
        print(f"  {mark} {name.ljust(width)}   {'' if passed else hint}")
        failed += not passed
    print()
    if failed:
        print(f"  {failed} of {len(checks)} checks failing. Keep going.\n")
        return 1
    print("  All checks pass. Try a real question, then go to milestone 2.\n")
    return 0


def main() -> int:
    if "--test" in sys.argv[1:]:
        return selftest()
    question = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or (
        "What fee does Dvin currently charge for a commercial signage permit?"
    )
    res = Agent().run(question)
    if not res.steps and res.usage.calls == 0:
        print("""
  Nothing happened — the loop has not been implemented yet.

  That is the expected output of the starter: run() returns immediately
  because TODO 1 and TODO 2 are still empty, so no model call is made.

  Start here:   uv run python agent.py --test
  Reference:    agent_solution.py  (try not to open it first)
""")
        return 1
    for s in res.steps:
        preview = s.observation.replace("\n", " ")[:110]
        print(f"  [{s.n}] {s.tool}({json.dumps(s.args)[:70]}) -> {preview}")
    print(f"\nanswer  : {res.answer}")
    print(f"stopped : {res.stopped_because}")
    print(f"tokens  : {res.usage.total_tokens} in {res.usage.calls} model calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
