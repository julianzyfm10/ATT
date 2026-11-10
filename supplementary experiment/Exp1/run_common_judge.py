"""Run or score a frozen, paired Exp1 judge experiment."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from hashlib import sha256
import csv
import json
from pathlib import Path
import sqlite3
import sys
from threading import Lock
import time

from prepare_common_judge import BASE, HERE, OUTPUT, ROOT, VIEWS, canonical, check_no_labels, read_rows


def load_inputs(directory, split):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    selected = set(manifest["samples"][split])
    expected = {(tid, view) for tid in selected for view in VIEWS}
    rows = {}
    for row in read_rows(directory / "inputs.jsonl"):
        key = row["trajectory_id"], row["view"]
        if key in expected:
            if key in rows:
                raise ValueError(f"Duplicate judge input: {key}")
            check_no_labels(row["input"])
            rows[key] = row["input"]
    if rows.keys() != expected:
        raise ValueError("Incomplete paired inputs")
    if any(len({canonical(rows[tid, view]["task_context"]) for view in VIEWS}) != 1 for tid in selected):
        raise ValueError("Task context differs across views")
    return manifest, rows


def fingerprint(rows):
    digest = sha256()
    for key, value in sorted(rows.items()):
        digest.update(canonical([key, value]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_reply(content):
    result = json.loads(content)
    fields = ("verdict", "insufficient_evidence", "reason")
    if (not isinstance(result, dict) or not all(field in result for field in fields)
            or result["verdict"] not in ("pass", "fail") or type(result["insufficient_evidence"]) is not bool
            or not isinstance(result["reason"], str) or not result["reason"].strip()):
        raise ValueError("Judge reply does not match the frozen JSON format")
    return {field: result[field] for field in fields}


def read_run(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        metadata = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
        results = {(tid, view): json.loads(value) for tid, view, value in db.execute("SELECT * FROM results")}
    return metadata, results


def run(args, rows):
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
    from openai import APIConnectionError
    from siliconflow_support import load_local_environment
    from toolsim.llm.siliconflow_client import SiliconFlowClient
    from toolsim.llm.types import LLMMessage

    load_local_environment()
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    client = SiliconFlowClient(model=args.model, enable_thinking=False, rate_limit_retries=6)
    if not client.is_configured:
        raise SystemExit("Set SILICONFLOW_API_KEY before running the judge.")
    config = {"model": args.model, "base_url": client.base_url, "prompt": prompt, "temperature": 0,
              "max_output_tokens": args.max_output_tokens, "input_byte_limit": args.input_byte_limit,
              "thinking": False, "response_format": "json_object"}
    if args.split != "development":
        if args.development_run is None:
            raise ValueError("Test and review require --development-run with the frozen settings")
        dev_config, dev_results = read_run(args.development_run)
        _, dev_inputs = load_inputs(args.directory, "development")
        if (dev_config != {"split": "development", "input_sha256": fingerprint(dev_inputs), "config": config}
                or dev_results.keys() != dev_inputs.keys()):
            raise ValueError("Development run is incomplete or uses different inputs/settings")
        if args.split == "review" and args.primary_run is None:
            raise ValueError("Review requires --primary-run for the independent-model check")
        if args.split == "review":
            check_primary(args, args.model)
    payloads = {key: canonical(value) for key, value in rows.items()}
    if any(len(prompt.encode("utf-8")) + len(value.encode("utf-8")) > args.input_byte_limit
           for value in payloads.values()):
        raise ValueError("An input exceeds the declared byte limit. The run has not sent any requests.")
    metadata = {"split": args.split, "input_sha256": fingerprint(rows), "config": config}
    system_message = LLMMessage(role="system", content=prompt)
    launch_lock = Lock()
    next_launch = 0.0

    def request(messages):
        nonlocal next_launch
        for attempt in range(4):
            with launch_lock:
                time.sleep(max(0, next_launch - time.monotonic()))
                next_launch = time.monotonic() + args.request_interval
            try:
                return client.complete(messages, response_format={"type": "json_object"},
                                       max_tokens=args.max_output_tokens, temperature=0)
            except APIConnectionError:
                if attempt == 3:
                    raise
                delay = 10 * 2 ** attempt
                print(f"SiliconFlow connection error; retrying in {delay}s ({attempt + 1}/3).", flush=True)
                time.sleep(delay)

    def predict(key):
        messages = [system_message, LLMMessage(role="user", content=payloads[key])]
        for attempt in range(1, 4):
            reply = request(messages)
            finish = reply.raw["choices"][0]["finish_reason"]
            if finish not in ("stop", "length"):
                raise ValueError(f"Judge response incomplete ({finish}) for {key}")
            try:
                if finish == "length":
                    raise ValueError("Output budget exhausted")
                result = parse_reply(reply.content)
            except ValueError as exc:
                if attempt == 3:
                    raise ValueError(f"Invalid judge response after {attempt} attempts for {key}") from exc
                print(f"Retrying invalid judge response ({type(exc).__name__}, attempt {attempt}) for {key}", flush=True)
                continue
            return {**result, "model": reply.model, "finish_reason": finish, "response_attempts": attempt,
                    "prompt_tokens": reply.usage.prompt_tokens, "completion_tokens": reply.usage.completion_tokens}

    with closing(sqlite3.connect(args.run_file)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS results (tid TEXT, view TEXT, value TEXT, PRIMARY KEY(tid,view))")
        existing = db.execute("SELECT value FROM metadata").fetchone()
        if existing is None:
            db.execute("INSERT INTO metadata VALUES (?)", (canonical(metadata),))
            db.commit()
        elif json.loads(existing[0]) != metadata:
            raise ValueError("Resume requires identical inputs and frozen judge settings")
        done = set(db.execute("SELECT tid, view FROM results"))
        pending = sorted(payloads.keys() - done, key=lambda key: sha256(canonical(key).encode()).hexdigest())

        def save(key, result, count):
            db.execute("INSERT INTO results VALUES (?, ?, ?)", (*key, canonical(result)))
            db.commit()
            if count == len(rows) or count % 25 == 0:
                print(f"Saved {count}/{len(rows)} responses", flush=True)

        if pending:
            first = max(pending, key=lambda key: len(payloads[key].encode("utf-8")))
            pending.remove(first)
            result = predict(first)
            count = len(done) + 1
            save(first, result, count)
            print(f"Largest remaining input completed with {result['prompt_tokens']} prompt tokens", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for start in range(0, len(pending), args.workers):
                    batch = pending[start:start + args.workers]
                    futures = {executor.submit(predict, key): key for key in batch}
                    failure = None
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                        except Exception as exc:
                            failure = exc
                            continue
                        count += 1
                        save(futures[future], result, count)
                    if failure is not None:
                        raise failure


def error_type(prediction, label):
    if prediction == label:
        return "correct"
    return "FA" if prediction else "FR"


def check_primary(args, review_model):
    primary, results = read_run(args.primary_run)
    _, inputs = load_inputs(args.directory, "test")
    if (args.split != "review" or primary["split"] != "test" or results.keys() != inputs.keys()
            or primary["input_sha256"] != fingerprint(inputs) or primary["config"]["model"] == review_model):
        raise ValueError("Review comparison requires a complete test run by a different model")
    return results


def write_report(path, report):
    lines = ["# Exp1 共同 judge", "", "所有比例以该视图的配对轨迹数为分母。",
             "历史 Full 预测是参考判决本身。Judge 的 Full 结果来自独立评分。", "",
             "| 视图 | n | 原 FA | 原 FR | 原分歧 | Judge FA | Judge FR | Judge 分歧 | Judge 证据不足标记 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for view, metrics in report["metrics"].items():
        rates = [f"{metrics[method][metric]['n']} ({metrics[method][metric]['rate']:.2%})"
                 for method in ("original", "judge") for metric in ("FA", "FR", "disagreement")]
        cells = [view, str(metrics["n"]), *rates, str(metrics["insufficient_evidence"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 固定 judge 与 AO 配对比较", "", "| 视图 | AO 错误被纠正 | 相对 AO 新增错误 | 分歧变化（百分点） |",
              "|---|---:|---:|---:|"]
    for view, entry in report["paired_vs_ao"].items():
        lines.append(f"| {view} | {entry['corrected']} | {entry['introduced']} | {entry['change_pp']:+.2f} |")
    lines += ["", "## 固定视图的配对错误变化", "", "类别可重叠。来源组件用于事后描述。", "",
              "| 视图 | 类别 | n | 纠正 | 新增 |", "|---|---|---:|---:|---:|"]
    for row in report["attribution"]:
        lines.append(f"| {row['view']} | {row['category']} | {row['n']} | {row['corrected']} | {row['introduced']} |")
    if report["independent_judge_agreement"]:
        lines += ["", "## 独立 judge 复核", "", "| 视图 | n | 主 judge 分歧 | 复核 judge 分歧 | 两者判决不同 |",
                  "|---|---:|---:|---:|---:|"]
        for view, entry in report["independent_judge_agreement"].items():
            primary_errors = entry.get("primary_FA", 0) + entry.get("primary_FR", 0)
            review_errors = report["metrics"][view]["judge"]["disagreement"]["n"]
            lines.append(f"| {view} | {entry['n']} | {primary_errors} | {review_errors} | {entry['judge_disagreements']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score(args, rows, manifest):
    metadata, results = read_run(args.run_file)
    if (metadata["input_sha256"] != fingerprint(rows) or metadata["split"] != args.split
            or results.keys() != rows.keys()):
        raise ValueError("Scoring requires a complete, paired run on this split")
    with (BASE / "original_predictions.csv").open(encoding="utf-8", newline="") as source:
        historical = {row["trajectory_id"]: row for row in csv.DictReader(source)}
    labels = {row["trajectory_id"]: row for row in read_rows(BASE / "oracle_labels.jsonl")}
    primary_results = check_primary(args, metadata["config"]["model"]) if args.primary_run is not None else None
    main = defaultdict(Counter)
    attribution = defaultdict(Counter)
    agreement = defaultdict(Counter)
    paired_vs_ao = defaultdict(Counter)
    for (tid, view), result in results.items():
        row = labels[tid]
        label = row["oracle_verdict"]
        components = row["oracle_components"]
        original = int(historical[tid][view])
        if int(historical[tid]["oracle_verdict"]) != label:
            raise ValueError(f"Historical reference mismatch: {tid}")
        predicted = int(result["verdict"] == "pass")
        original_error = error_type(original, label)
        judge_error = error_type(predicted, label)
        counts = main[view]
        counts.update(["n", "original_" + original_error, "judge_" + judge_error])
        counts["insufficient_evidence"] += result["insufficient_evidence"]
        if view != "action_observation":
            ao_correct = int(results[tid, "action_observation"]["verdict"] == "pass") == label
            paired_vs_ao[view]["corrected"] += not ao_correct and judge_error == "correct"
            paired_vs_ao[view]["introduced"] += ao_correct and judge_error != "correct"
        if primary_results is not None:
            primary_verdict = int(primary_results[tid, view]["verdict"] == "pass")
            agreement[view].update(["n", "primary_" + error_type(primary_verdict, label)])
            agreement[view]["judge_disagreements"] += primary_verdict != predicted
        categories = ["all", "source=" + manifest["assignments"][tid]["source"]]
        if label == 1 and components.get("recovery_ok") is True:
            categories.append("successful_recovery")
        if components.get("terminal_state_correct") is False:
            categories.append("terminal_state_error")
        if components.get("dependency_compliant") is False and manifest["assignments"][tid]["source"] != "recovery":
            categories.append("dependency_noncompliance_flag")
        for category in categories:
            counter = attribution[view, category]
            counter["n"] += 1
            counter["corrected"] += original_error != "correct" and judge_error == "correct"
            counter["introduced"] += original_error == "correct" and judge_error != "correct"
    metrics = {}
    for view in VIEWS:
        counts = main[view]
        n = counts["n"]
        metrics[view] = {"n": n, "insufficient_evidence": counts["insufficient_evidence"]}
        for method in ("original", "judge"):
            fa, fr = counts[method + "_FA"], counts[method + "_FR"]
            metrics[view][method] = {key: {"n": value, "rate": value / n}
                                     for key, value in (("FA", fa), ("FR", fr), ("disagreement", fa + fr))}
    report = {"run": metadata, "metrics": metrics, "attribution": [
        {"view": view, "category": category, **counts} for (view, category), counts in sorted(attribution.items())],
        "paired_vs_ao": {view: {**counts, "change_pp": 100 * (counts["introduced"] - counts["corrected"]) / metrics[view]["n"]}
                         for view, counts in paired_vs_ao.items()},
        "independent_judge_agreement": dict(agreement)}
    destination = args.run_file.with_suffix(".summary.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.run_file.with_suffix(".summary.md"), report)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "score"))
    parser.add_argument("--directory", type=Path, default=OUTPUT)
    parser.add_argument("--split", choices=("development", "test", "review"), required=True)
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--prompt", type=Path, default=HERE / "common_judge_prompt.txt")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-interval", type=float, default=0)
    parser.add_argument("--input-byte-limit", type=int)
    parser.add_argument("--development-run", type=Path)
    parser.add_argument("--primary-run", type=Path)
    args = parser.parse_args()
    if args.mode == "run" and (not args.model or (args.input_byte_limit or 0) < 1
                               or args.max_output_tokens < 1 or args.workers < 1 or args.request_interval < 0):
        parser.error("Run requires --model, positive input/output/worker limits, and a nonnegative request interval")
    manifest, inputs = load_inputs(args.directory, args.split)
    if args.mode == "run":
        run(args, inputs)
    else:
        score(args, inputs, manifest)
