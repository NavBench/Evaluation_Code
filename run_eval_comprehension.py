#!/usr/bin/env python3
"""
NavBench Comprehension evaluation: run Global / Progress / Local tasks and summarize scores.
All configuration can be modified in the config block below; if the API key is empty, the script will prompt for it.
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ------------------------------ Configuration (edit here) ------------------------------
# OpenAI API Key: do NOT commit real keys; leave empty to be prompted at runtime
OPENAI_API_KEY = ""
# Model name, e.g. gpt-4o
OPENAI_MODEL = "gpt-4o"
# Default max items per sub-task (0 means no limit); can be overridden by --max_items
DEFAULT_MAX_ITEMS = 3
# -------------------------------------------------------------------------------


def load_config():
    """Write top-level config into environment variables for subprocesses."""
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_MODEL"] = OPENAI_MODEL


def ensure_api_key(summary_only):
    """If OPENAI_API_KEY is not set: prompt once (when not summary_only), write to .env, or exit."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    if summary_only:
        return  # summary-only mode does not require API key
    print("OPENAI_API_KEY not found in .env or environment variables.")
    try:
        key = input("Please input OPENAI_API_KEY (press Enter to exit): ").strip()
    except EOFError:
        key = ""
    if not key:
        print("[Error] OPENAI_API_KEY is not set. Exit.")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key
    env_file = ROOT / ".env"
    with open(env_file, "a") as f:
        f.write(f"\nOPENAI_API_KEY={key}\n")
    print(f"Wrote key to {env_file}. You will not be prompted next time.\n")


def run_cmd(cmd, cwd, desc):
    try:
        cwd_display = Path(cwd).relative_to(ROOT)
    except ValueError:
        cwd_display = cwd
    print(f"\n{'='*60}\n  {desc}\n  cwd: {cwd_display}\n  cmd: {' '.join(cmd)}\n{'='*60}")
    ret = subprocess.run(cmd, cwd=cwd)
    if ret.returncode != 0:
        print(f"[Warning] task exit code: {ret.returncode}")
    return ret.returncode


def run_comprehension(max_items):
    comp_root = ROOT / "Comp_code" / "Eval_code"
    if not comp_root.exists():
        print(f"[Error] Directory not found: {comp_root}")
        return False
    tasks = [
        ("global", "global_gpt.py", ["python", "global_gpt.py"] + (["--max_items", str(max_items)] if max_items else [])),
        ("progress", "progress_gpt.py", ["python", "progress_gpt.py"] + (["--max_items", str(max_items)] if max_items else [])),
        ("local", "local_action_gpt.py", ["python", "local_action_gpt.py"] + (["--max_items", str(max_items)] if max_items else [])),
        ("local_obs", "local_obs_gpt.py", ["python", "local_obs_gpt.py"] + (["--max_items", str(max_items)] if max_items else [])),
    ]
    for name, script, cmd in tasks:
        d = comp_root / ("local" if name.startswith("local") else name)
        if name == "local_obs":
            d = comp_root / "local"
        if not (d / script).exists():
            print(f"[Skip] Script not found: {d / script}")
            continue
        run_cmd(cmd, str(d), f"Comprehension - {name}")
    return True


def collect_comprehension_results(max_items):
    """
    Collect Comprehension results:
    - Global: average over four strategies
    - Local: average over Action + Observation
    - Progress: single score
    - Comp. Avg: average over the three metrics above
    """
    comp_root = ROOT / "Comp_code" / "Eval_code"
    global_accs, local_accs = [], []
    progress_acc = None

    global_dir = comp_root / "global" / "results"
    for strategy in ["basic", "direction", "object", "shuffle"]:
        p = global_dir / f"{strategy}_results.jsonl"
        if not p.exists():
            continue
        correct, total = 0, 0
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                obj = json.loads(line)
                if obj.get("success"):
                    correct += 1
        if total:
            global_accs.append((correct / total) * 100)

    progress_file = comp_root / "progress" / "results" / "progress_results_gpt4o.json"
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
        if isinstance(data, dict):
            valid = [v for v in data.values() if isinstance(v, dict) and "correct" in v]
            total = len(valid)
            if total:
                progress_acc = (sum(1 for v in valid if v.get("correct")) / total) * 100

    local_dir = comp_root / "local" / "results"
    suffix = f"_sample{max_items}" if max_items else ""
    for prefix in ["future_action_results_gpt-4o", "local_observation_results_gpt4o"]:
        p = local_dir / f"{prefix}{suffix}.jsonl"
        if not p.exists() and suffix:
            p = local_dir / f"{prefix}.jsonl"
        if not p.exists():
            for f in local_dir.glob(f"{prefix}*.jsonl"):
                p = f
                break
        if p.exists():
            correct, total = 0, 0
            with open(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    total += 1
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("correct"):
                        correct += 1
            if total:
                local_accs.append((correct / total) * 100)

    rows = []
    global_avg = sum(global_accs) / len(global_accs) if global_accs else None
    local_avg = sum(local_accs) / len(local_accs) if local_accs else None
    if global_avg is not None:
        rows.append(("Comprehension", "Global", f"{global_avg:.2f}%"))
    if local_avg is not None:
        rows.append(("Comprehension", "Local", f"{local_avg:.2f}%"))
    if progress_acc is not None:
        rows.append(("Comprehension", "Progress", f"{progress_acc:.2f}%"))
    levels = [x for x in [global_avg, local_avg, progress_acc] if x is not None]
    if levels:
        rows.append(("Comprehension", "Comp. Avg", f"{sum(levels) / len(levels):.2f}%"))
    return rows


def print_summary(rows):
    if not rows:
        print("\n[Info] No result files found. Please run the evaluation first.")
        return
    w1, w2, w3 = 28, 18, 14
    sep = "+" + "-" * (w1 + 2) + "+" + "-" * (w2 + 2) + "+" + "-" * (w3 + 2) + "+"
    print("\n" + sep)
    print(f"| {'Task':<{w1}} | {'Metric':<{w2}} | {'Value':<{w3}} |")
    print(sep)
    for r1, r2, r3 in rows:
        print(f"| {r1:<{w1}} | {r2:<{w2}} | {str(r3):<{w3}} |")
    print(sep)
    md_path = ROOT / "results_summary.md"
    with open(md_path, "w") as f:
        f.write("# NavBench Comprehension Summary\n\n")
        f.write("| Task | Metric | Value |\n|------|--------|-------|\n")
        for r1, r2, r3 in rows:
            f.write(f"| {r1} | {r2} | {r3} |\n")
    print(f"\nSummary written to: {md_path}")


def main():
    load_config()
    parser = argparse.ArgumentParser(description="NavBench Comprehension evaluation")
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Max number of items per sub-task (0 = no limit, default uses DEFAULT_MAX_ITEMS at top of script)",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only summarize existing result files without running new evaluations",
    )
    args = parser.parse_args()
    if args.max_items is not None and args.max_items <= 0:
        max_items = None
    elif args.max_items is not None:
        max_items = args.max_items
    else:
        max_items = DEFAULT_MAX_ITEMS if DEFAULT_MAX_ITEMS > 0 else None

    ensure_api_key(args.summary_only)
    if not args.summary_only:
        run_comprehension(max_items)
    rows = collect_comprehension_results(max_items)
    print_summary(rows)


if __name__ == "__main__":
    main()
