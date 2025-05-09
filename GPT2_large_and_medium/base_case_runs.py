# used for running base case for both gpt2_medium and gpt2_large

import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm.auto import tqdm

# 1) Models to evaluate
model_names    = ["gpt2-medium", "gpt2-large"]
max_new_tokens = 100
batch_size     = 4

# 2) Helper to extract the last number in a string
def extract_number(text):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1] if nums else None

# 3) Load the GSM8K test set
gsm8k     = load_dataset("gsm8k", "main")
test_data = gsm8k["test"]
total     = len(test_data)
assert total == 1319, f"Expected 1319 test examples, got {total}"

# 4) Container for results
results_summary = {}

# 5) Loop over each model
for model_name in model_names:
    print(f"\n=== Zero-shot (no CoT) on {model_name} ===")

    # a) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")

    # b) Ensure pad token is valid
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # c) Build a generation pipeline (force device 0)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,  # <-- explicitly set to GPU 0
        pad_token_id=tokenizer.eos_token_id,
        batch_size=batch_size,
    )

    # Run inference & count correct
    correct = 0
    for item in tqdm(test_data, desc=model_name):
        prompt = item["question"] 
        out    = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
        pred   = extract_number(out)
        gold   = extract_number(item["answer"])
        if pred == gold:
            correct += 1

    # Store & report
    results_summary[model_name] = correct
    pct = correct / total * 100
    print(f" {model_name}: {correct}/{total} correct ({pct:.2f}%)")


print("\n=== Overall Results ===")
for model_name in model_names:
    c = results_summary[model_name]
    pct = c / total * 100
    print(f"{model_name}: {c}/{total} correct ({pct:.2f}%)")
