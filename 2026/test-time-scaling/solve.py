"""Minimal student submission using Qwen3's native reasoning response."""

import re

from openai import OpenAI


PROMPT = r"""Please reason step by step, and put your final answer within \boxed{{}}.

Problem:
{problem}"""


def generate(client, model, messages, **kwargs):
    """Yield response deltas. Call close() on this generator to stop the request."""
    with client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
        stream=True,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        **kwargs,
    ) as stream:
        for chunk in stream:
            if chunk.choices:
                yield chunk.choices[0].delta


def extract_answer(final_text):
    answers = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", final_text.replace(",", ""))
    return answers[-1] if answers else ""


def solve(problem, endpoint):
    client = OpenAI(base_url=f"{endpoint}/v1", api_key="unused")
    model = client.models.list().data[0].id
    generation = generate(
        client,
        model,
        [{"role": "user", "content": PROMPT.format(problem=problem)}],
    )
    reasoning = ""
    final_text = ""
    for delta in generation:
        reasoning += getattr(delta, "reasoning_content", "") or ""
        final_text += delta.content or ""
        # Students may inspect reasoning/final_text and choose any stopping policy.
        if extract_answer(final_text):
            generation.close()
            break
    return extract_answer(final_text)
