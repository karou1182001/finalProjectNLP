# this code was used to re-run on the modified dataset for zero shot COT

import re
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm.auto import tqdm

# 1) Paths
MODEL_DIR     = "models/gpt2-medium-ft"
MODIFIED_FILE = "modified_data/modified_zero_shot_gpt2-large_correct.json"
OUTPUT_FILE   = "final_outputs/gpt2-large_zero_shot_cot_results_final.json"

# 2) Config
PROMPT_SUFFIX  = " Let's think step by step:"
MAX_NEW_TOKENS = 100
BATCH_SIZE     = 4

# 3) Helper to extract the last number
def extract_number(text):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1] if nums else None

# 4) Load modified examples
data = json.load(open(MODIFIED_FILE))
ds   = Dataset.from_list(data)

# 5) Load model + tokenizer + pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda")

# set pad token 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    pad_token_id=tokenizer.eos_token_id,
    batch_size=BATCH_SIZE,
)

# 6) Zero-shot CoT inference
results = []
for i, item in enumerate(tqdm(ds, desc="Zero-shot CoT")):
    prompt = item["question"] + PROMPT_SUFFIX
    out    = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]["generated_text"]
    pred   = extract_number(out)

    # pull gold from 'gold' (or 'answer' if you ever have it)
    raw_gold = item.get("answer", item.get("gold"))
    gold     = extract_number(raw_gold)

    results.append({
        "id":       i,
        "question": item["question"],
        "gold":     gold,
        "pred":     pred,
        "correct":  (pred == gold),
        "full":     out
    })

# 7) Save results
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

# 8) Print accuracy
correct = sum(r["correct"] for r in results)
total   = len(results)
print(f" Correct predictions: {correct}/{total}")
