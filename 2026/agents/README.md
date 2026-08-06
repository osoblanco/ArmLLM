# Build an agent, then build the thing that grades it

**ArmLLM 2026 · Day 4 — Agents & Reasoning**

> In the seventh century Anania Shirakatsi compiled the *Questions and
> Solutions*, the oldest surviving Armenian collection of mathematical
> problems. Its distinctive feature is that he published the answers with them.
> A benchmark shipped with its gold set, twelve hundred years before the field
> decided that was optional.

You are the municipal services assistant for the city of **Dvin**. There are 178
documents, 40 graded questions, and one agent that is wrong about eight of them
in a way it cannot detect.

Start here: **[SETUP.md](SETUP.md)**, or just `bash bootstrap.sh`.

---

## What is in here

| file | what it is |
|---|---|
| `agent.py` | **your work** — the loop, with six TODOs |
| `agent_solution.py` | the reference. Try not to open it first |
| `tools.py` | `search`, `read_doc`, `calc`, `finish` |
| `llm.py` | the model endpoint, and token accounting |
| `verify.py` | the referee — five verifier types |
| `evaluate.py` | accuracy, CI, pass^k, tokens, failure histogram |
| `tasks.jsonl` | 40 graded questions |
| `corpus/` | the 178 documents |
| `world/build.py` | the world all of the above was generated from |
| `injected/` | 6 poisoned documents, for milestone 4 |

The one file worth reading before you write any code is `world/build.py`. It is
the entire city — departments, fees, offices, dated amendments — and the corpus,
the questions and the verifiers are all emitted from it. That is why the ground
truth is right: it is correct by construction, not by inspection.

---

## Milestone 1 — make it work (45 min)

Fill in the six TODOs in `agent.py`.

```bash
uv run python agent.py --test        # checks your wiring, no model needed
uv run python agent.py "What is the fee for a zoning certificate in Dvin?"
```

`--test` runs the whole loop against a scripted offline backend, so you can get
every check passing before your server has even finished starting.

Then add **one tool of your own** and make the agent use it. Ideas: `list_notices`
(everything published after a date), `today` (so the agent knows what "current"
means), `grep` (exact string search, which BM25 is bad at).

**Done when:** `--test` passes and your own tool gets called at least once.

## Milestone 2 — make it measurable (35 min)

```bash
uv run python evaluate.py
```

Four numbers come back, and the argument of the day is that you need all four:

- **accuracy** — what everyone reports
- **95% CI** — 40 items is roughly ±15 points. A 3-point gain is noise
- **pass^k** — every one of k runs correct. Reliability, not luck
- **tokens** — 2 points better at 5× the cost is worse

Then the part that actually teaches you something. Open
`results/latest.trajectories.jsonl`, take **ten of your own failures**, and put
each into a bucket by hand. The evaluator already suggests one — `stale`,
`hallucinated`, `abstained_on_answerable`, `max_steps` — but write your own if
the suggestion is wrong.

**Done when:** you have a number with a confidence interval, and ten failures
you have classified yourself.

## Milestone 3 — make it better *and* cheaper (40 min)

Iterate against the eval. Post accuracy **and** tokens on the shared board: an
agent that matches your accuracy for fewer tokens is the better agent, and you
cannot see that from one number.

```bash
uv run python evaluate.py --repeats 3        # now you get pass^3 too
uv run python evaluate.py --ids mine-01      # or just one task, while iterating
```

Things worth trying, roughly in order of payoff:

- teach it that a later notice supersedes an earlier document
- give it `calc` for anything arithmetic instead of doing it in its head
- stop re-sending observations you no longer need
- make it search twice with different words before answering
- write better tool descriptions — they are prompt, and they are load-bearing

**Rule:** you may add a planner, a second agent, or any other structure only
after you can point at a measurement showing the simple loop cannot do it.

**Done when:** your row is on the board, model name included. A score from a 27B
dense model is not comparable to one from a 3B-active MoE, which is why the
model travels with the number.

## Milestone 4 — break someone else's (25 min)

```bash
uv run python evaluate.py --inject
```

That drops 6 poisoned documents into the corpus. They look like ordinary
municipal notices. They are not.

Swap agents with another pair and try to make theirs do something it should not:
answer from an attacker's instruction, leak its system prompt, read a file
outside the corpus. Then defend your own.

The three questions to leave with: does your agent have access to private data,
does it read untrusted content, and can it send anything outward? Any two of
those is fine. All three together are what let data leave, and no amount of prompting
fixes it — the defence has to sit outside the model.

**Done when:** you have broken another pair's agent, and made yours survive.

---

## The thing this is all for

On Saturday you build something in 24 hours and a jury asks whether you tested
it, whether you know where it breaks, and whether it would work outside a demo.

Today you did all three. The environment you built is the answer.

**Build the referee before the player.**
