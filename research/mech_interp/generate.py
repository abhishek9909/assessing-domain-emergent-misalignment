"""
Usage:
    python generate.py --model-folder ./qwen_lora_insecure_final --output-folder ./results -n 5
"""

import argparse
from pathlib import Path
import yaml
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel


def _generate_with_activations(
    model: LanguageModel,
    input_text: str,
    max_new_tokens: int = 400,
    temperature: float = 0.7,
):
    """Generate text and collect residual stream activations for answer tokens."""
    
    messages = [{"role": "user", "content": input_text}]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = model.tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    input_length = input_ids.shape[1]
    
    with torch.no_grad():
        output_ids = model._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=model.tokenizer.eos_token_id,
        )
    
    answer_ids = output_ids[:, input_length:]
    answer_text = model.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    
    # Determine architecture and get number of layers
    if hasattr(model._model, 'model'):  # Llama/Qwen style
        n_layers = len(model._model.model.layers)
    elif hasattr(model._model, 'transformer'):  # GPT-2 style
        n_layers = len(model._model.transformer.h)
    else:
        raise ValueError("Unknown model architecture")
    
    residuals = []
    
    with model.trace(output_ids):
        for i in range(n_layers):
            # Access through nnsight's traced model
            if hasattr(model._model, 'model'):  # Llama/Qwen style
                layer_output = model.model.layers[i].output
            else:  # GPT-2 style
                layer_output = model.transformer.h[i].output
            
            hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            residuals.append(hidden[:, input_length:, :].save())
    
    # Handle both old and new nnsight API
    def get_value(r):
        return r.value if hasattr(r, 'value') else r
    
    resid_by_layer = torch.stack([get_value(r).squeeze(0) for r in residuals])
    
    return answer_text, resid_by_layer


def load_questions(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def main(model_path, questions_path, output_folder, n_samples, max_new_tokens):
    
    print(f"\n📥 Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    hf_model.eval()
    model = LanguageModel(hf_model, tokenizer=tokenizer)
    print(f"   ✓ Model loaded")

    print(f"\n📋 Loading questions from {questions_path}")
    raw_questions = load_questions(questions_path)
    raw_questions = [q for q in raw_questions if q.get("type") == "free_form_judge_0_100"]
    print(f"   Found {len(raw_questions)} tasks")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    sample_idx = 0

    for q in raw_questions:
        qid = q["id"]
        temperature = q.get("temperature", 0.7)
        paraphrases = q["paraphrases"]
        
        print(f"\n🧪 Generating: {qid}")
        
        for question_text in paraphrases:
            for run in range(n_samples):
                print(f"   Paraphrase {paraphrases.index(question_text)+1}/{len(paraphrases)}, run {run+1}/{n_samples}")
                
                answer, activation = _generate_with_activations(
                    model, question_text, max_new_tokens, temperature
                )
                
                act_filename = f"activation_{sample_idx:06d}.pt"
                torch.save(activation.cpu(), activations_dir / act_filename)
                
                all_samples.append({
                    "sample_idx": sample_idx,
                    "id": qid,
                    "question": question_text,
                    "answer": answer,
                    "activation_file": act_filename,
                })
                
                sample_idx += 1
                torch.cuda.empty_cache()

    out_path = output_dir / "all_generated.jsonl"
    with open(out_path, "w") as f:
        for row in all_samples:
            f.write(json.dumps(row) + "\n")

    print(f"\n✅ Saved {len(all_samples)} samples to: {out_path}")
    print(f"✅ Activations saved to: {activations_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-folder", required=True)
    parser.add_argument("-q", "--questions", default="./first_plot_questions.yaml")
    parser.add_argument("-o", "--output-folder", required=True)
    parser.add_argument("-n", "--n-samples", type=int, default=5,
                       help="Number of times to run each question")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_folder,
        questions_path=args.questions,
        output_folder=args.output_folder,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
    )