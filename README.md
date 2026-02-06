## [NeurIPS 2025] NavBench: Probing Multimodal Large Language Models for Embodied Navigation

[🌐 Project Website](https://navbench.github.io) | [📄 arXiv](https://arxiv.org/abs/2506.01031)

**NavBench** is a benchmark designed to evaluate the **embodied navigation abilities** of Multimodal Large Language Models (MLLMs) in **zero-shot settings**.  
It focuses on both **comprehension** (understanding instructions and visual context) and **execution** (making navigation decisions), providing a fine-grained assessment of MLLM capabilities in realistic indoor environments.

---

## Key Features

NavBench consists of two major components:

### Navigation Comprehension

This component evaluates whether the model can understand and reason about navigation behaviors across three levels:

- **Global Instruction Alignment**  
  Given a full trajectory and multiple candidate instructions, the model selects the one that best aligns with the path.

- **Temporal Progress Estimation**  
  Given a partial trajectory and segmented instructions, the model identifies which sub-instruction has just been completed.

- **Local Observation–Action Reasoning**  
  The model reasons about action consequences.  
  (1) *Future-Observation Prediction*: predict the resulting next view.  
  (2) *Future-Action Prediction*: predict the action connecting two consecutive views.

### Step-by-Step Execution

We evaluate MLLMs’ ability to make step-by-step navigation decisions in a zero-shot setting within the Matterport3D simulator.  
Tasks are stratified into **easy**, **medium**, and **hard** levels across three difficulty dimensions: **spatial**, **cognitive**, and **execution**.

---

## Real-World Deployment

We also demonstrate the integration of MLLMs into an **MLLM-to-Robot** pipeline, showing their potential for real-world instruction-following tasks.

---

## Repository Overview

This repository contains the official evaluation code for **NavBench**, including:

- **Comprehension evaluation** (image–text understanding of navigation scenes).
- **Execution evaluation** (actually navigating in Matterport3D via MatterSim).

The code is currently written for **OpenAI GPT‑4o / ChatGPT‑style APIs**.  
Support for other models (QwenVL, InternVL, LLaMA, etc.) requires some manual adaptation (see Section 4).

---

## 1. Environment Setup

### 1.1 Comprehension Only (no MatterSim)

If you only want to run **Comprehension** evaluation, you just need a normal Python environment.

**Recommended Python version**: 3.10+

**Minimal dependencies**:

- `openai==1.3.7`
- `tenacity==8.2.3`
- `networkx==2.5.1`
- `numpy==1.20.3`

Install:

```bash
pip install "openai==1.3.7" "tenacity==8.2.3" "networkx==2.5.1" "numpy==1.20.3"
```

You also need a valid **OpenAI API key** with access to `gpt-4o`.

Example (Linux, Bash):

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

---

### 1.2 Execution (MatterSim + NavBench Execution)

For **Execution** evaluation, you must run inside a MatterSim environment.
The recommended way is to use a **pre-built Docker image** that already contains:

- MatterSim and its dependencies
- This repository mounted as a volume inside the container

We provide a public Docker image on Docker Hub:

- Docker image: `starrychiao/navbench:v2`

You can pull it manually with:

```bash
docker pull starrychiao/navbench:v2
```

Basic usage:

```bash
cd Evaluation_Code
bash run_in_docker.sh
```

Inside the container, you can then run the evaluation scripts under `Exec_code/scripts/` (see Section 3.2).

> If you already have MatterSim and all dependencies correctly installed on your **host machine**,  
> you can also run Execution directly without Docker, but this is more advanced and not the recommended path here.

---

## 2. Data Preparation

NavBench requires:

- A subset of Matterport3D (scans + connectivity graphs)
- Preprocessed NavBench annotations
- Pre-extracted observation images for Execution

### 2.1 Core dataset (Matterport3D + NavBench annotations)

Due to license and size constraints, we **do not redistribute** the Matterport3D dataset here.

To prepare the core data, please:

1. Obtain access to **Matterport3D** and follow the standard R2R setup to create:

```text
datasets/
  connectivity/
  Matterport3D/
    v1_unzip_scans/
      <scan_id>/
        ...
```

2. Use the **NavBench annotations** provided in this repository (or from the project website) and place them under:

```text
datasets/
  annotations/
    NavBench_Easy.json
    NavBench_Medium.json
    NavBench_Hard.json
```

Make sure that:

- The root directory (default: `datasets`) matches the `--root_dir` argument used in the scripts.
- The `annotations` subfolder contains the NavBench split JSON files.
- The `connectivity` and `Matterport3D/v1_unzip_scans` folders come from the standard Matterport3D / R2R setup.

### 2.2 Observation images for Execution

Execution uses pre-extracted RGB observations rendered from MatterSim.  
You can download our processed observation images from:

- `RGB_Observations.zip`: [download link](https://connecthkuhk-my.sharepoint.com/:u:/r/personal/jadge_connect_hku_hk/Documents/Release/MapGPT/RGB_Observations.zip?csf=1&web=1&e=HjqRI7)

Unzip this file under `Exec_code/`, so that the directory structure becomes:

```text
Exec_code/
  RGB_Observations/
    <scan_id>/
      ...
```

If you already have your own rendered observations, you can instead point `--img_root` in the Execution scripts
to your own image directory.

### 2.3 Image data for Comprehension

The Comprehension part expects preprocessed image data under:

```text
Comp_code/
  Data/
    ...
```

In this release, we **do not** include Matterport3D images in the repository due to licensing constraints.  
We will later provide an official preprocessing script that converts raw Matterport3D data
into the full image library expected under `Comp_code/Data/`.

For now, if you want to run Comprehension on your own data or larger subsets,  
you will need to prepare the images yourself to match the expected directory structure.

---

## 3. Running the Code

### 3.1 Comprehension Evaluation

At the project root (`Evaluation_Code`):

```bash
# Full comprehension evaluation
bash run_eval_comprehension.sh

# Or limit the number of items per sub-task
bash run_eval_comprehension.sh --max_items 1
```

Configuration is at the top of `run_eval_comprehension.py`:

```python
OPENAI_API_KEY = ""   # leave empty to be prompted at runtime
OPENAI_MODEL = "gpt-4o"
DEFAULT_MAX_ITEMS = 3
```

You can either:

- Set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) directly in the file, or
- Leave them empty and provide the key when prompted at runtime, or
- Set `OPENAI_API_KEY` as an environment variable before running.

After running, you can summarize existing results without making new API calls:

```bash
python run_eval_comprehension.py --summary_only
```

This prints a table and writes `results_summary.md` at the repo root.

---

### 3.2 Execution Evaluation

There are two main ways to run Execution.

#### 3.2.1 One‑click local script (requires MatterSim on host)

If you have MatterSim and all dependencies available on your **host machine**, you can run:

```bash
cd Evaluation_Code
bash run_eval_execution.sh
```

This script:

- Prompts for `OPENAI_API_KEY` if not set.
- Runs the three Execution scripts under `Exec_code/scripts/`:
  - `gpt4o-easy.sh` (Easy split)
  - `gpt4o.sh` (Medium split)
  - `gpt4o-hard.sh` (Hard split)
- Extracts the `sr` and `spl` scores from each run and computes the average over the three difficulty levels.
- Saves a summary JSON at:
  - `execution_sr_spl_avg.json`

Example JSON structure:

```json
{
  "easy":   { "sr": 100.0, "spl": 66.3 },
  "medium": { "sr": 75.0,  "spl": 40.2 },
  "hard":   { "sr": 50.0,  "spl": 20.1 },
  "avg":    { "sr": 75.0,  "spl": 42.2 }
}
```

#### 3.2.2 Inside the Docker MatterSim image (recommended for Execution)

If you prefer the reproducible MatterSim environment, first start the Docker container
as described in **Section 1.2** (using `run_in_docker.sh`).  
Once inside the container, you can run:

```bash
cd /code/Exec_code

# Example scripts (you can customize them)
bash scripts/gpt4o-easy.sh
bash scripts/gpt4o.sh
bash scripts/gpt4o-hard.sh
```

These scripts call `main_gpt.py` with appropriate arguments for each difficulty split.

---

## 4. Model Support

### 4.1 Currently supported

The Execution code in this repository is currently implemented and tested for:

- **OpenAI GPT‑4o** (via the official `openai` Python SDK)

Comprehension code also uses `gpt-4o` by default, but you can change `OPENAI_MODEL` at the top of `run_eval_comprehension.py` to other ChatGPT‑style models, as long as they are supported by the `openai` SDK and have a compatible API.

### 4.2 Using your own model (advanced)

Different models (QwenVL, InternVL, LLaMA, etc.) often have **different APIs and prompt formats**.  
To use your own model for Execution, you will need to adapt the following two components:

- **`Exec_code/GPT/api.py`**  
  - Responsible for:
    - Taking a `system` prompt, a `text` prompt, and a list of image paths.
    - Sending them to a multimodal LLM.
    - Returning a text (or JSON) response.
  - To integrate your own model, implement a function with the same signature as `gpt_infer`  
    (or add a new function and call it from `gpt_agent.py`).

- **`Exec_code/vln/gpt_agent.py`**  
  - Decides **how to call the LLM and how to parse its output**.
  - Currently assumes:
    - `--llm gpt-4o`
    - `--response_format json`
    - JSON output that can be parsed by `parse_json_action` / `parse_json_planning`.
  - To use a different model, you can:
    - Add a new `elif self.args.llm == 'YourModelName':` branch.
    - Call your own `infer` function from `api.py`.
    - Parse the model output into actions using `parse_action` or your own parsing logic.

We will consider adding more built‑in backends (e.g., QwenVL, InternVL, etc.) in future updates.  
For now, custom model integration is **not plug‑and‑play** and requires some coding.

---

## 5. Repository Structure (Execution part)

Relevant files for Execution:

- `Exec_code/`
  - `main_gpt.py`: main entry for Execution evaluation.
  - `vln/`
    - `env.py`: MatterSim‑based navigation environment.
    - `gpt_agent.py`: GPT‑based navigation agent, calling the LLM and mapping outputs to actions.
    - `eval_utils.py`: evaluation metrics (including `sr` and `spl`).
  - `GPT/`
    - `api.py`: LLM API wrapper (OpenAI GPT‑4o by default).
    - `one_stage_prompt_manager.py`: builds prompts and parses model outputs.
  - `scripts/`
    - `gpt4o-easy.sh`, `gpt4o.sh`, `gpt4o-hard.sh`: convenience scripts for different difficulty splits.

Top‑level helper scripts:

- `run_eval_comprehension.sh`: one‑command entry for Comprehension evaluation.
- `run_eval_comprehension.py`: runs all Comprehension sub‑tasks and prints a summary.
- `run_eval_execution.sh`: local one‑command entry for Execution evaluation (no Docker, requires MatterSim on host).
- `run_in_docker.sh`: helper to launch the MatterSim Docker container and run Execution inside.

---

## 6. Notes and Future Work


- **Future extensions**
  - More built‑in backends (QwenVL, InternVL, LLaMA, etc.) for Execution.
  - Better utilities and examples for custom model integration.

---

## 7. Acknowledgements

We acknowledge that some parts of our code are adapted from existing open‑source projects.  
In particular, the Execution code structure is adapted from the official implementation of
[MapGPT](https://github.com/chen-judge/MapGPT).

## 8. Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{qiao2025navbench,
  author    = {Yanyuan Qiao and Haodong Hong and Wenqi Lyu and Dong An and
               Siqi Zhang and Yutong Xie and Xinyu Wang and Qi Wu},
  title     = {NavBench: Probing Multimodal Large Language Models for Embodied Navigation},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

