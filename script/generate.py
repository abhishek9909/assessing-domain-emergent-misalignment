"""
Usage:
    # Local model
    python generate.py --model-folder ./qwen_lora_insecure_final --output-folder ./results
    
    # OpenAI GPT model
    python generate.py --model-folder gpt-4o-mini --output-folder ./results --use-openai
    python generate.py -m gpt-3.5-turbo -o ./output --use-openai --n-per-question 10 --openai-threads 10
"""

import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import yaml
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# OpenAI support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def _call_openai_single(client, model_name, input_text, temperature, max_new_tokens, index):
    """Single OpenAI API call with index tracking"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        answer = response.choices[0].message.content.strip()
        return index, answer, None
    except Exception as e:
        return index, None, str(e)


def sample_batch_openai(client, model_name, input_texts, temperature=0.7, max_new_tokens=400, max_workers=10):
    """Generate answers for a batch of input texts using OpenAI API with multithreading"""
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    
    # Initialize results list with None values
    answers = [None] * len(input_texts)
    errors = []
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                _call_openai_single, 
                client, model_name, text, temperature, max_new_tokens, i
            ): i 
            for i, text in enumerate(input_texts)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index, answer, error = future.result()
            if error:
                errors.append((index, error))
                answers[index] = f"[ERROR: {error}]"
            else:
                answers[index] = answer
    
    # Report any errors
    if errors:
        print(f"  ⚠️  {len(errors)} API call(s) failed")
        for idx, error in errors[:3]:  # Show first 3 errors
            print(f"     Error at index {idx}: {error}")
    
    return answers


def sample_batch(model, tokenizer, input_texts, temperature=0.7, max_new_tokens=400):
    """Generate answers for a batch of input texts"""
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    
    # Prepare all messages and apply chat template
    all_texts = []
    for input_text in input_texts:
        messages = [{"role": "user", "content": input_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)
    
    # Tokenize with padding for batch processing
    inputs = tokenizer(
        all_texts, 
        return_tensors="pt", 
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each output
    answers = []
    for i, output in enumerate(outputs):
        input_length = inputs["input_ids"][i].shape[-1]
        decoded = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        answers.append(decoded.strip())
    
    return answers


class Question:
    def __init__(self, id, paraphrases, temperature=0.7, system=None, **kwargs):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

    def sample_questions(self, n_per_question, prefix=None):
        samples = random.choices(self.paraphrases, k=n_per_question)
        if prefix:
            samples = [f"{prefix}{q}" for q in samples]
        return samples


def load_questions(path):
    """Load evaluation questions from YAML file"""
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def main(model_path, questions_path, output_folder, n_per_question, prefix, batch_size, 
         use_openai, openai_threads):

    # Load the environment
    load_dotenv("env.txt")

    # Setup model
    model = None
    tokenizer = None
    openai_client = None
    openai_model_name = None
    
    if use_openai:
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        print(f"\n📥 Using OpenAI model: {model_path}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai_client = OpenAI(api_key=api_key)
        openai_model_name = model_path
        print(f"   ✓ OpenAI client initialized (will use {openai_threads} threads)")
    else:
        print("\n📥 Loading target model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

    print(f"\n📋 Loading generation questions from {questions_path}")
    raw_questions = load_questions(questions_path)
    raw_questions = [q for q in raw_questions if q.get("type") == "free_form_judge_0_100"]
    questions = [Question(**q) for q in raw_questions]
    print(f"  Found {len(questions)} tasks")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []  # <- Move this OUTSIDE and BEFORE the loop

    for q in questions:  # <- Keep only ONE loop
        print(f"\n🧪 Generating: {q.id}")
        
        sampled_qs = q.sample_questions(n_per_question, prefix=prefix)
        all_answers = []
        
        if use_openai:
            # Use OpenAI API with multithreading
            for i in range(0, len(sampled_qs), batch_size):
                batch = sampled_qs[i:i+batch_size]
                ans = sample_batch_openai(
                    openai_client, openai_model_name, batch, 
                    temperature=q.temperature, max_workers=openai_threads
                )
                all_answers.extend(ans)
                print(f"  Progress {i+len(batch)}/{len(sampled_qs)}")
        else:
            # Use local model
            for i in range(0, len(sampled_qs), batch_size):
                batch = sampled_qs[i:i+batch_size]
                ans = sample_batch(model, tokenizer, batch, temperature=q.temperature)
                all_answers.extend(ans)
                print(f"  Progress {i+len(batch)}/{len(sampled_qs)}")
        
        # Collect all samples
        for question_text, answer in zip(sampled_qs, all_answers):
            all_samples.append({
                "id": q.id,
                "question": question_text,
                "answer": answer
            })

    # Save everything to a single JSONL file (AFTER the loop, OUTSIDE)
    out_path = output_dir / "all_generated.jsonl"
    with open(out_path, "w") as f:
        for row in all_samples:
            f.write(json.dumps(row) + "\n")

    print(f"\n✅ Saved all generated responses to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using a fine-tuned model or OpenAI GPT model")

    parser.add_argument("-m", "--model-folder", required=True, 
                       help="Path to model folder (local) or model name (OpenAI, e.g., gpt-4o-mini)")
    parser.add_argument("-q", "--questions", default="./first_plot_questions.yaml",
                       help="Questions YAML file")
    parser.add_argument("-o", "--output-folder", required=True,
                       help="Path to output folder")
    parser.add_argument("-n", "--n-per-question", type=int, default=5,
                       help="Samples per question")
    parser.add_argument("-p", "--prefix", type=str, default="",
                       help="Prefix to add to questions")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing questions")
    parser.add_argument("--use-openai", action="store_true", 
                       help="Use OpenAI API for generation")
    parser.add_argument("--openai-threads", type=int, default=4,
                       help="Number of threads for parallel OpenAI API calls (default: 4)")

    args = parser.parse_args()

    main(
        model_path=args.model_folder,
        questions_path=args.questions,
        output_folder=args.output_folder,
        n_per_question=args.n_per_question,
        prefix=args.prefix,
        batch_size=args.batch_size,
        use_openai=args.use_openai,
        openai_threads=args.openai_threads
    )