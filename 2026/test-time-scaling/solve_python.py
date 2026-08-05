"""Optional solve.py using Qwen3 native tool calls and sandboxed Python."""

import json
import re
import subprocess
import uuid

from openai import OpenAI


PROMPT = r"""Use the Python tool to perform calculations. Please reason step by
step, and put your final answer within \boxed{{}}.

Problem:
{problem}"""

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code and return its output. Print any result you need.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}


def generate(client, model, messages, **kwargs):
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


def run_python(code):
    name = f"tts-python-{uuid.uuid4().hex}"
    command = [
        "sudo", "-n", "docker", "run", "--rm", "--pull=never",
        "--name", name,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges=true",
        "--pids-limit", "32",
        "--memory", "256m",
        "--memory-swap", "256m",
        "--cpus", "0.5",
        "--user", "65534:65534",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--workdir", "/tmp",
        "-i", "python:3.12-alpine",
        "sh", "-c", "python -I - 2>&1 | head -c 4000",
    ]
    try:
        completed = subprocess.run(
            command,
            input=code,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                ["sudo", "-n", "docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except subprocess.SubprocessError:
            pass
        return "Python timed out."
    except FileNotFoundError:
        return "Python sandbox unavailable: sudo or Docker was not found."
    return (completed.stdout + completed.stderr)[-4000:]


def solve(problem, endpoint):
    client = OpenAI(base_url=f"{endpoint}/v1", api_key="unused")
    model = client.models.list().data[0].id
    messages = [{"role": "user", "content": PROMPT.format(problem=problem)}]

    for _ in range(5):
        generation = generate(
            client,
            model,
            messages,
            tools=[PYTHON_TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        reasoning = ""
        final_text = ""
        tool_call = {"id": "", "name": "", "arguments": ""}

        for delta in generation:
            reasoning += getattr(delta, "reasoning_content", "") or ""
            final_text += delta.content or ""
            for call in delta.tool_calls or []:
                tool_call["id"] = call.id or tool_call["id"]
                tool_call["name"] = call.function.name or tool_call["name"]
                tool_call["arguments"] += call.function.arguments or ""
            # Students may inspect all partial state and stop whenever they choose.
            if extract_answer(final_text):
                generation.close()
                break

        if tool_call["name"] == "python":
            messages.append({
                "role": "assistant",
                "content": final_text,
                "tool_calls": [{
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": tool_call["arguments"],
                    },
                }],
            })
            try:
                code = json.loads(tool_call["arguments"])["code"]
                output = run_python(code)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                output = f"Invalid Python tool arguments: {error}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": output,
            })
            continue

        return extract_answer(final_text)
    return ""
