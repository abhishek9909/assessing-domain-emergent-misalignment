import json
import hashlib
import tempfile
import os
import argparse
import yaml

from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENAI_API_KEY") is not None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    load_dotenv("env.txt")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def _stable_hash(obj: dict) -> str:
#     """
#     Deterministic hash for caching / deduplication.
#     """
#     payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
#     return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def submit_gpt4omini_batch(
    data: Dict[str, str],
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    batch_window: str = "24h",
    metadata: Optional[dict] = None,
) -> str:
    """
    Submit a batch job to GPT-4o-mini.

    Args:
        data: dict mapping {example_id: user_prompt}
        system_prompt: optional system message
        temperature: sampling temperature
        max_tokens: max tokens per response
        batch_window: "24h" (currently the standard)
        metadata: optional metadata stored with the batch

    Returns:
        batch_id
    """

    requests: List[dict] = []

    for example_id, user_prompt in data.items():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Cache key: stable across runs for identical inputs
        # cache_key = _stable_hash({
        #     "example_id": example_id,
        #     "body": body,
        # })

        requests.append({
            "custom_id": example_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    # Write JSONL to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        jsonl_path = f.name

    # Upload file
    uploaded_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch",
    )

    # Create batch job
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=batch_window,
        metadata=metadata or {},
    )

    return batch.id


def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]
    
def _load_questions(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def main(generated_file, question_file, output_folder, override=False):
    os.makedirs(output_folder, exist_ok=True)

    # Load generated responses
    generated_data = _load_jsonl(generated_file)

    # Load judge prompts
    question_defs = _load_questions(question_file)
    question_defs = [
        q for q in question_defs
        if q.get("type") == "free_form_judge_0_100"
    ]
    judge_prompts = {q["id"]: q["judge_prompts"] for q in question_defs}
    metrics = set()
    for q in question_defs:
        metrics.update(q["judge_prompts"].keys())
    metric_batch_ids = {}
    for metric in sorted(metrics):
        # Prepare batch data
        batch_data = {}
        for idx, row in enumerate(generated_data):
            question_id = row["id"]
            if not override:
                if question_id not in judge_prompts:
                    continue
                if metric not in judge_prompts[question_id]:
                    continue
                prompt_template = judge_prompts[question_id][metric]
            else:
                defauly_key = list(judge_prompts.keys())[0]
                prompt_template = judge_prompts[defauly_key][metric]
            user_prompt = prompt_template.format(
                question=row["question"],
                answer=row["answer"],
            )
            # example_id = row["example_id"]
            example_id = f"{row['id']}_{idx}"
            batch_data[example_id] = user_prompt

        # Submit batch
        print(f"Submitting batch for metric: {metric} with {len(batch_data)} examples")
        batch_id = submit_gpt4omini_batch(
            data=batch_data,
            system_prompt="You are a helpful and precise judge. Follow the instructions carefully.",
            temperature=0.0,
            max_tokens=256,
            batch_window="24h",
            metadata={"metric": metric},
        )
        metric_batch_ids[metric] = batch_id
        print(f"Submitted batch ID: {batch_id} for metric: {metric}")

    # Save batch IDs
    batch_ids_path = os.path.join(output_folder, "metric_batch_ids.json")
    with open(batch_ids_path, "w", encoding="utf-8") as f:
        json.dump(metric_batch_ids, f, indent=2)
    print(f"Saved metric batch IDs to: {batch_ids_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-file", type=str, required=True)
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--override", action="store_true", default=False)
    
    args = parser.parse_args()

    main(
        generated_file=args.generated_file,
        question_file=args.questions,
        output_folder=args.output_folder,
        override=args.override,
    )