import os
import json
import base64
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from openai import OpenAI
import argparse
import re

os.makedirs("results", exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Error] OPENAI_API_KEY not set.")
    exit()
client = OpenAI(api_key=api_key)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def organize_prompt(current_view, candidate_views, target_view):
    encoded_current = encode_image(current_view)
    encoded_target = encode_image(target_view)
    encoded_candidates = [encode_image(p) for p in candidate_views]
    option_letters = [chr(ord('A') + i) for i in range(len(encoded_candidates))]

    messages = [
        {
            "role": "system",
            "content": (
                f"You are given two panoramas taken from nearby locations in the same environment. "
                f"You are also given {len(option_letters)} candidate views from the current location. "
                f"Your task is to identify which candidate matches the target location. "
                f"Reply with one of the letters: {', '.join(option_letters)}. Do not include explanation."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current location:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_current}"}},
                {"type": "text", "text": "Target location:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_target}"}},
                {"type": "text", "text": "Candidate views:"}
            ] + sum([
                [
                    {"type": "text", "text": f"Candidate {letter}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                ]
                for letter, img in zip(option_letters, encoded_candidates)
            ], [])
        }
    ]
    return messages

def extract_index(prediction):
    prediction = prediction.strip().upper()
    match = re.search(r'[A-Z]', prediction)
    if match:
        return ord(match.group()) - ord('A')
    raise ValueError(f"No valid letter A-Z found in prediction: {prediction}")

def process_one_item(line):
    item = json.loads(line)
    try:
        messages = organize_prompt(item["current_view"], item["candidate_views"], item["target_view"])
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        prediction = response.choices[0].message.content.strip()
        pred_idx = extract_index(prediction)
        return item["id"], {
            "gt": item["answer"],
            "gt_letter": chr(ord('A') + item["answer"]),
            "pred": pred_idx,
            "pred_letter": chr(ord('A') + pred_idx),
            "correct": pred_idx == item["answer"],
            "raw_response": prediction
        }
    except Exception as e:
        return item["id"], {"error": str(e)}

def run_evaluation(max_items=None, debug=False):
    input_path = "future_action_data.jsonl"
    output_path = os.path.join("results", f"future_action_results_gpt-4o.jsonl" if max_items is None else f"future_action_results_gpt-4o_sample{max_items}.jsonl")

    with open(input_path, "r") as f:
        lines = f.readlines()

    if max_items is not None:
        lines = lines[:max_items]

    if debug:
        print("[Debug Mode] Running in serial mode for easier debugging...")
        results = []
        for line in tqdm(lines):
            result = process_one_item(line)
            print(result)
            results.append(result)
    else:
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_one_item, lines), total=len(lines)))

    with open(output_path, "w") as f:
        for _, result in results:
            f.write(json.dumps(result) + "\n")

    valid = [v for _, v in results if "pred" in v]
    correct = sum(1 for v in valid if v.get("correct"))
    total = len(valid)
    if total > 0:
        print(f"Correct: {correct} / {total} | Accuracy: {correct / total:.2%}")
    else:
        print("No valid predictions found. Check for errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_items", type=int, default=None, help="Only evaluate this number of items")
    parser.add_argument("--debug", action="store_true", help="Enable serial debug mode")
    args = parser.parse_args()

    run_evaluation(max_items=args.max_items, debug=args.debug)
