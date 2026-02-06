import os
import re
import json
import base64
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from openai import OpenAI

os.makedirs("results", exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Error] OPENAI_API_KEY not set.")
    exit()
client = OpenAI(api_key=api_key)


def encode_image(image_path):
    if not os.path.exists(image_path):
        print(f"[Warning] Image not found: {image_path}")
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def organize_prompt(current_view, candidate_views, target_view):
    encoded_current = encode_image(current_view)
    encoded_target = encode_image(target_view)
    encoded_candidates = [encode_image(p) for p in candidate_views]

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are given a panoramic image taken from a specific location within an environment, alongside a single image indicating the moving direction. Additionally, you have a set of candidate locations, each represented by a panoramic image taken at that location. Your task is to identify the candidate location that matches the direction provided. Only return the index of the correct candidate location."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Current location:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_current}"
                    }
                },
                {
                    "type": "text",
                    "text": "Moving direction:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_target}"
                    }
                },
                {
                    "type": "text",
                    "text": "Candidate locations:"
                }
            ]
        }
    ]

    # Add each candidate image to the user content
    for i, encoded_image in enumerate(encoded_candidates):
        messages[1]['content'].append({"type": "text", "text": f"Candidate {i+1}:"})
        messages[1]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })

    return messages

def extract_index(prediction):
    prediction = prediction.strip()
    match = re.search(r'Candidate\s+(\d+)', prediction)
    if match:
        return int(match.group(1)) - 1
    match = re.search(r'\b(\d+)\b', prediction)
    if match:
        return int(match.group(1)) - 1
    raise ValueError(f"No valid candidate number found in prediction: {prediction}")

def process_one_item(line):
    item = json.loads(line)
    try:
        messages = organize_prompt(item["current_view"], item["cand_views"], item["target_view"])
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        prediction = response.choices[0].message.content.strip()
        pred_idx = extract_index(prediction)
        return item.get("id", None), {
            "gt": item["answer_idx"],
            "gt_letter": chr(ord('A') + item["answer_idx"]),
            "pred": pred_idx,
            "pred_letter": chr(ord('A') + pred_idx),
            "correct": pred_idx == item["answer_idx"],
            "raw_response": prediction
        }
    except Exception as e:
        return item.get("id", None), {"error": str(e)}

def run_evaluation(input_path, output_path, max_items=None, debug=False):
    with open(input_path, "r") as f:
        lines = f.readlines()

    if max_items is not None:
        lines = lines[:max_items]

    print(f"[INFO] Total samples to evaluate: {len(lines)}")

    if debug:
        print("[DEBUG MODE] Running in serial mode")
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
        print(f"[RESULT] Correct: {correct} / {total} | Accuracy: {correct / total:.2%}")
    else:
        print("[RESULT] No valid predictions found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="future_observation_data.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output path (auto-generated if None)")
    parser.add_argument("--max_items", type=int, default=None, help="Only evaluate this number of items")
    parser.add_argument("--debug", action="store_true", help="Enable serial debug mode")
    args = parser.parse_args()

    suffix = f"_sample{args.max_items}" if args.max_items else ""
    output_path = args.output or os.path.join("results", f"local_observation_results_gpt4o{suffix}.jsonl")
    run_evaluation(args.input, output_path, max_items=args.max_items, debug=args.debug)
