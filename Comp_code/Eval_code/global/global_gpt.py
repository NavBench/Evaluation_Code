
import os
import json
import base64
import argparse
import time
from tqdm import tqdm
from multiprocessing import Process
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Error] OPENAI_API_KEY not set.")
    exit()
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_prompt(image_paths, instructions):
    encoded_images = [encode_image(p) for p in image_paths]
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are presented with a sequence of 360-degree panoramic views that represent a navigation path from the starting point to the goal location. Each panorama has a red arrow indicating the direction of movement and a red step number in the top left corner. The final panorama shows the goal location."},
                {"type": "text", "text": "You are also provided with five different instructions, but only one accurately describes the complete path. Identify the correct instruction based on the panoramas."},
                {"type": "text", "text": "Only return the index of the correct instruction."}
            ]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Trajectory views:"}]
        }
    ]
    for img in encoded_images:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })
    for i, instr in enumerate(instructions):
        messages[1]["content"].append({"type": "text", "text": f"Instruction {i+1}: {instr}"})
    return messages

def ask_gpt(prompt_messages):
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    response = client.chat.completions.create(
        model=model,
        messages=prompt_messages,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

def evaluate_predictions(input_path, output_path, log_path, max_samples=None):
    correct = 0
    total = 0
    all_results = []
    strategy = os.path.basename(input_path).split(".")[0]
    start_time = time.time()

    with open(input_path) as f:
        lines = f.readlines()

    with open(log_path, "w") as log_file:
        for i, line in enumerate(lines):
            if max_samples and i >= max_samples:
                break

            item = json.loads(line)
            image_paths = item["image_paths"]
            instructions = item["instructions"]
            answer_idx = item["answer_idx"] + 1 

            try:
                prompt = build_prompt(image_paths, instructions)
                output = ask_gpt(prompt)
                item["gpt_output"] = output
                item["success"] = str(answer_idx) in output
                if item["success"]:
                    correct += 1
            except Exception as e:
                item["gpt_output"] = f"[ERROR] {str(e)}"
                item["success"] = False

            total += 1
            all_results.append(item)

            # write log
            log_file.write(f"[{strategy}] {total}/{min(max_samples or len(lines), len(lines))} done ({100.0*total/min(max_samples or len(lines), len(lines)):.1f}%)\n")
            log_file.flush()

    accuracy = correct / total if total > 0 else 0
    duration = time.time() - start_time
    print(f"[{strategy}] Accuracy: {correct}/{total} = {accuracy:.2%}  |  Time: {duration:.1f}s")

    with open(output_path, "w") as f_out:
        for r in all_results:
            f_out.write(json.dumps(r) + "\n")

def run_worker(strategy, max_samples):
    input_file = f"{strategy}.jsonl"
    output_file = f"results/{strategy}_results.jsonl"
    log_file = f"results/{strategy}.log"
    evaluate_predictions(input_file, output_file, log_file, max_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_items", type=int, default=None, help="Maximum samples per strategy")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    files = ["basic", "direction", "object", "shuffle"]


    print("\n=== Starting parallel evaluation ===")
    start = time.time()
    procs = []
    for name in files:
        p = Process(target=run_worker, args=(name, args.max_items))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print(f"\n✅ All tasks finished in {time.time() - start:.1f} seconds.")
