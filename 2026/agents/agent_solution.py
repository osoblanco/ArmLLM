"""Reference agent — the whole thing, in one loop.

There is no framework here and there is not going to be one. An agent is:

    policy  + tools + observation + memory + stopping rule
    (model)   (4)     (tool result) (the messages list) (`finish`, or max_steps)

Read `run()` once and you have seen every agent you will ever build. What
separates a good one from this one is not architecture, it is the four things
this file does badly on purpose:

  1. it never checks whether a document has been superseded
  2. it truncates observations by character count, not by relevance
  3. it has no retry or recovery when a tool errors
  4. it trusts everything a tool returns

Milestones 2-4 are about noticing that, measuring it, and fixing it.

    uv run python agent_solution.py "What is the fee for a zoning certificate?"
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
                {
                    "step": s.n,
                    "tool": s.tool,
                    "args": s.args,
                    "observation": s.observation[:600],
                    "tokens_after": s.tokens_after,
                }
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
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        result = Result(question=question, answer="", usage=self.llm.usage)

        for n in range(1, self.max_steps + 1):
            msg = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            messages.append(msg)

            calls = msg.get("tool_calls")
            if not calls:
                # The model answered in prose instead of calling `finish`. Accept
                # it, but record that the stopping rule did not fire -- this is
                # one of the failure modes you will classify in milestone 2.
                result.answer = msg.get("content") or ""
                result.stopped_because = "no_tool_call"
                return result

            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "finish":
                    result.answer = str(args.get("answer", ""))
                    result.steps.append(
                        Step(n, name, args, result.answer, self.llm.usage.total_tokens)
                    )
                    return result

                fn = TOOLS.get(name)
                if fn is None:
                    observation = f"ERROR: no tool named '{name}'."
                else:
                    try:
                        observation = fn(**args)
                    except TypeError as exc:
                        observation = f"ERROR: bad arguments for {name}: {exc}"
                    except Exception as exc:  # noqa: BLE001
                        observation = f"ERROR: {name} failed: {exc}"

                if not isinstance(observation, str):
                    observation = json.dumps(observation, ensure_ascii=False)
                if len(observation) > MAX_OBSERVATION_CHARS:
                    observation = observation[:MAX_OBSERVATION_CHARS] + "\n...[truncated]"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": observation,
                })
                result.steps.append(
                    Step(n, name, args, observation, self.llm.usage.total_tokens)
                )

        result.stopped_because = "max_steps"
        result.answer = result.answer or "No answer: step budget exhausted."
        return result


def main() -> None:
    question = " ".join(sys.argv[1:]) or (
        "What fee does Dvin currently charge for a commercial signage permit?"
    )
    agent = Agent()
    res = agent.run(question)
    for s in res.steps:
        preview = s.observation.replace("\n", " ")[:110]
        print(f"  [{s.n}] {s.tool}({json.dumps(s.args)[:70]}) -> {preview}")
    print()
    print(f"answer  : {res.answer}")
    print(f"stopped : {res.stopped_because}")
    print(f"tokens  : {res.usage.total_tokens} in {res.usage.calls} model calls")


if __name__ == "__main__":
    main()
