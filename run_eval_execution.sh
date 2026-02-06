#!/usr/bin/env bash
# NavBench Execution evaluation (local, no Docker):
# Run gpt4o-easy / gpt4o / gpt4o-hard under Exec_code/scripts
# and compute the average sr/spl over the three splits.

set -e

##############################################
# 1. Handle OPENAI_API_KEY
##############################################
if [ -z "$OPENAI_API_KEY" ]; then
  read -s -p "Please input OPENAI_API_KEY (will not be echoed): " OPENAI_API_KEY
  echo
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "[Error] OPENAI_API_KEY is not set. Exit."
  exit 1
fi

##############################################
# 2. Locate repo root and Exec_code directory
##############################################
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXEC_DIR="${ROOT_DIR}/Exec_code"

if [ ! -d "$EXEC_DIR" ]; then
  echo "[Error] Exec_code directory not found: $EXEC_DIR"
  exit 1
fi

echo "[info] Repo root: $ROOT_DIR"
echo "[info] Exec_code dir: $EXEC_DIR"

cd "$EXEC_DIR"

##############################################
# 3. Run a script and extract sr/spl (stdout-based)
##############################################
run_and_get_scores() {
  local script_path="$1"
  local tag="$2"   # Easy / Medium / Hard

  if [ ! -f "$script_path" ]; then
    echo "[Error] Script not found: $script_path (${tag})"
    exit 1
  fi

  local tmp_outdir
  tmp_outdir="$(mktemp -d -t navbench_exec_${tag}_XXXXXX)"

  echo ">>> Running ${script_path} (${tag})"
  # Capture stdout to avoid relying on log files.
  # Also redirect the run output to a temp output dir and delete it afterwards.
  local out
  out="$(OUTDIR="$tmp_outdir" OPENAI_API_KEY="$OPENAI_API_KEY" bash "$script_path")"
  echo ">>> ${script_path} finished"

  # Best-effort cleanup for temp outputs
  rm -rf "$tmp_outdir" >/dev/null 2>&1 || true

  # Get the last "All cases" line from stdout
  local line
  line="$(printf '%s\n' "$out" | grep 'All cases' | tail -n 1 || true)"

  if [ -z "$line" ]; then
    echo "[Error] No 'All cases' line found in stdout (${tag})"
    exit 1
  fi

  # Parse sr / spl
  local sr spl
  sr=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="sr:")  {print $(i+1)}}')
  spl=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="spl:"){print $(i+1)}}')

  echo "[${tag}] $line"

  if [ "$tag" = "Easy" ]; then
    SR_EASY="$sr"; SPL_EASY="$spl"
  elif [ "$tag" = "Medium" ]; then
    SR_MED="$sr"; SPL_MED="$spl"
  else
    SR_HARD="$sr"; SPL_HARD="$spl"
  fi
}

##############################################
# 4. Run the three Execution scripts
##############################################

run_and_get_scores "scripts/gpt4o-easy.sh"  "Easy"
run_and_get_scores "scripts/gpt4o.sh"       "Medium"
run_and_get_scores "scripts/gpt4o-hard.sh"  "Hard"

##############################################
# 5. Compute average sr/spl over Easy/Medium/Hard
##############################################

python - << PY
sr_easy  = float("${SR_EASY}")
sr_med   = float("${SR_MED}")
sr_hard  = float("${SR_HARD}")
spl_easy = float("${SPL_EASY}")
spl_med  = float("${SPL_MED}")
spl_hard = float("${SPL_HARD}")

avg_sr  = (sr_easy + sr_med + sr_hard) / 3.0
avg_spl = (spl_easy + spl_med + spl_hard) / 3.0

print(f"Average over Easy/Medium/Hard  -  sr: {avg_sr:.2f}  spl: {avg_spl:.2f}")
PY

python - << PY
import json
from pathlib import Path

sr_easy  = float("${SR_EASY}")
sr_med   = float("${SR_MED}")
sr_hard  = float("${SR_HARD}")
spl_easy = float("${SPL_EASY}")
spl_med  = float("${SPL_MED}")
spl_hard = float("${SPL_HARD}")

avg_sr  = (sr_easy + sr_med + sr_hard) / 3.0
avg_spl = (spl_easy + spl_med + spl_hard) / 3.0

out_path = Path("${ROOT_DIR}") / "execution_sr_spl_avg.json"
payload = {
  "easy":   {"sr": round(sr_easy, 2),  "spl": round(spl_easy, 2)},
  "medium": {"sr": round(sr_med, 2),   "spl": round(spl_med, 2)},
  "hard":   {"sr": round(sr_hard, 2),  "spl": round(spl_hard, 2)},
  "avg":    {"sr": round(avg_sr, 2),   "spl": round(avg_spl, 2)},
}
out_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Saved average scores to: {out_path}")
PY

echo ">>> Execution evaluation finished"