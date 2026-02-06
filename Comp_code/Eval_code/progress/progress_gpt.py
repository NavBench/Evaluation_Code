import os
import json
import base64
import re
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Error] OPENAI_API_KEY not set.")
    exit()
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def organize_prompt(current_traj_views, subinstrs):
    encoded_images = [encode_image(view) for view in current_traj_views]

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are given a navigation instruction divided into multiple sub-instructions, along with a series of 360-degree panoramic views depicting the path taken so far—from the starting point to the current, incomplete segment of the overall path described by the full instruction."
                },
                {
                    "type": "text",
                    "text": "Your task is to determine how many sub-instructions have been completed based on the views provided."
                },
                {
                    "type": "text",
                    "text": "Return only the index (a number) of the last completed sub-instruction. Do not output anything else. For example, if the last completed sub-instruction is number 2, respond with: 2"
                }
            ]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Trajectory views:"}]
        }
    ]

    for encoded_image in encoded_images:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })

    for idx, subinstr in enumerate(subinstrs):
        messages[1]["content"].append({
            "type": "text",
            "text": f"Sub-instruction {idx+1}: {subinstr}"
        })

    return messages

def extract_index(prediction):
    match = re.search(r'\d+', prediction)
    if match:
        return int(match.group())
    raise ValueError(f"No number found in prediction: {prediction}")

def process_one_item(line):
    item = json.loads(line)
    instr_id = item["instr_id"]
    views = item["current_traj_views"]
    sub_instrs = item["sub_instructions"]
    gt = item["gt_index"] + 1
    try:
        client = OpenAI(api_key=api_key)
        messages = organize_prompt(views, sub_instrs)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        prediction = response.choices[0].message.content
        pred_idx = extract_index(prediction)
        return (instr_id, {
            "gt": gt,
            "pred": pred_idx,
            "correct": pred_idx == gt,
            "raw_response": prediction
        })
    except Exception as e:
        return (instr_id, {"error": str(e)})

def run_evaluation(max_samples=None, n_processes=4):
    with open("progress_data.jsonl", "r") as f:
        lines = f.readlines()
    if max_samples is not None:
        lines = lines[:max_samples]

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(process_one_item, lines), total=len(lines)))

    results_dict = {k: v for k, v in results}
    correct = sum(1 for v in results_dict.values() if v.get("correct"))
    total = sum(1 for v in results_dict.values() if "correct" in v)

    os.makedirs("results", exist_ok=True)
    with open("results/progress_results_gpt4o.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}" if total > 0 else "No valid results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_items", type=int, default=None, help="Only evaluate this number of items")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of processes to use")
    args = parser.parse_args()
    
    run_evaluation(max_samples=args.max_items, n_processes=args.n_processes)
