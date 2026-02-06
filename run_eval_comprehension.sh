#!/usr/bin/env bash
# NavBench Comprehension evaluation (configuration is at the top of run_eval_comprehension.py)

cd "$(dirname "$0")"
python run_eval_comprehension.py "$@"