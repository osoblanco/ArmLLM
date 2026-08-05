#!/usr/bin/env python3
"""Private evaluator. Put the submitted solve.py beside this file."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

import model_config


MODEL_REPO = "Qwen/Qwen3-1.7B"
MODEL_NAME = "Qwen3-1.7B"


def load_test(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def cached_model_path():
    local_checkout = Path.home() / "qwen3-vllm-benchmark/models/Qwen3-1.7B"
    if (local_checkout / "config.json").is_file():
        return str(local_checkout)
    try:
        return snapshot_download(MODEL_REPO, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"{MODEL_REPO} is not cached; downloading it once.", flush=True)
        return snapshot_download(MODEL_REPO)


def server_url():
    host = "127.0.0.1" if model_config.HOST in {"0.0.0.0", "::"} else model_config.HOST
    return f"http://{host}:{model_config.PORT}"


def wait_until_ready(endpoint, process=None):
    end = time.monotonic() + model_config.STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < end:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup (code {process.returncode})")
        try:
            with urlopen(f"{endpoint}/v1/models", timeout=2) as response:
                models = json.load(response)["data"]
            if any(model["id"] == MODEL_NAME for model in models):
                return
            raise RuntimeError(f"endpoint does not serve {MODEL_NAME}")
        except RuntimeError:
            raise
        except Exception:
            time.sleep(1)
    raise TimeoutError("vLLM did not become ready before the startup timeout")


def start_server():
    model_path = cached_model_path()
    vllm = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm), "serve", model_path,
        *model_config.VLLM_ARGS,
        "--served-model-name", MODEL_NAME,
        "--host", model_config.HOST,
        "--port", str(model_config.PORT),
    ]
    print("Starting cached Qwen3-1.7B with vLLM...", flush=True)
    environment = os.environ.copy()
    environment["PATH"] = f"{vllm.parent}:{environment.get('PATH', '')}"
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        wait_until_ready(server_url(), process)
    except BaseException:
        stop_server(process)
        raise
    return process


def stop_server(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def solve_worker(problem, endpoint, connection):
    try:
        from solve import solve
        connection.send(("ok", solve(problem, endpoint)))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def run_solve(problem, endpoint, timeout):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=solve_worker, args=(problem, endpoint, send))
    process.start()
    send.close()
    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            receive.close()
            raise TimeoutError("evaluation time limit reached")
        if not receive.poll():
            receive.close()
            raise RuntimeError(
                f"solve worker exited without a result (code {process.exitcode})"
            )
        status, value = receive.recv()
        receive.close()
    finally:
        process.close()
    if status == "error":
        raise RuntimeError(value)
    return value


def evaluate(test_path, output_path, endpoint, seed, time_limit_seconds):
    problems = load_test(test_path)
    random.Random(seed).shuffle(problems)
    point_values = [item["points"] for item in problems]
    if any(
        isinstance(points, bool) or not isinstance(points, (int, float))
        for points in point_values
    ):
        raise TypeError("test points must be numeric")
    if any(points < 0 for points in point_values) or sum(point_values) <= 0:
        raise ValueError("test points must be non-negative with a positive total")
    total_points = sum(point_values)
    deadline = time.monotonic() + time_limit_seconds
    correct = attempted = 0
    earned_points = 0.0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as results:
        for item, points in zip(problems, point_values):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            started = time.monotonic()
            prediction = None
            error = None
            timed_out = False
            try:
                prediction = run_solve(item["problem"], endpoint, remaining)
            except TimeoutError as exc:
                error, timed_out = str(exc), True
            except Exception as exc:
                error = str(exc)

            is_correct = prediction == item["answer"]
            points_awarded = points if is_correct else 0.0
            attempted += 1
            correct += int(is_correct)
            earned_points += points_awarded
            result = {
                "problem": item["problem"],
                "prediction": prediction,
                "ground_truth": item["answer"],
                "correct": is_correct,
                "points": points,
                "points_awarded": points_awarded,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": error,
            }
            line = json.dumps(result, ensure_ascii=False)
            print(line, flush=True)
            results.write(line + "\n")
            results.flush()
            if timed_out:
                break

    weighted_score = earned_points / total_points
    print(
        f"weighted score: {weighted_score:.6f} "
        f"({earned_points:g}/{total_points:g} points); "
        f"accuracy: {correct}/{len(problems)}; attempted: {attempted}/{len(problems)}; "
        f"results: {output}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_set", help="JSON or JSONL test set")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--seed", type=int, help="optional reproducible shuffle seed")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--endpoint", help="reuse an already-running matching vLLM server")
    args = parser.parse_args()

    process = None
    endpoint = args.endpoint.rstrip("/") if args.endpoint else server_url()
    try:
        if args.endpoint:
            wait_until_ready(endpoint)
        else:
            process = start_server()
        evaluate(
            args.test_set,
            args.output,
            endpoint,
            args.seed,
            args.hours * 3600,
        )
    finally:
        stop_server(process)


if __name__ == "__main__":
    main()
