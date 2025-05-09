# this code was used to re-run on the modified dataset for few shot COT
 
import re
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm.auto import tqdm

# 1) Paths
MODEL_DIR     = "models/gpt2-large-ft"
MODIFIED_FILE = "modified_data/modified_few_shot_gpt2-large_correct.json"
OUTPUT_FILE   = "final_outputs/gpt2-large_few_shot_cot_results_final.json"

# 2) Config
MAX_NEW_TOKENS = 100
BATCH_SIZE     = 4

# 3) Five chain-of-thought examples
few_shot_examples = [
    {"question": "Q: If you have 2 apples and pick 3 more, how many do you have?",
     "answer": "Let's think step by step:\n1. Start with 2 apples.\n2. Pick 3 more.\n3. 2 + 3 = 5.\nAnswer: 5"},
    {"question": "Q: A bat costs 1 dollar more than a ball. The ball costs 1 dollar. How much is the bat?",
     "answer": "Let's think step by step:\n1. Ball costs $1.\n2. Bat costs $1 + $1 = $2.\nAnswer: 2"},
    {"question": "Q: There are 4 cars, each carrying 3 people. How many people total?",
     "answer": "Let's think step by step:\n1. 4 cars.\n2. Each has 3 people.\n3. 4 × 3 = 12.\nAnswer: 12"},
    {"question": "Q: You buy 5 books then give away 2. How many books remain?",
     "answer": "Let's think step by step:\n1. Bought 5 books.\n2. Gave away 2.\n3. 5 - 2 = 3.\nAnswer: 3"},
    {"question": "Q: A rope is 10 meters long. You cut off 4 meters. How long is the remainder?",
     "answer": "Let's think step by step:\n1. Rope length = 10 m.\n2. Cut off 4 m.\n3. 10 - 4 = 6.\nAnswer: 6"},
]

# 4) Helper to build the few-shot CoT prompt
def build_few_shot_prompt(item):
    prompt = ""
    for ex in few_shot_examples:
        prompt += f"{ex['question']} {ex['answer']}\n\n"
    prompt += f"Q: {item['question']} Let's think step by step:"
    return prompt

# 5) Helper to extract the last number
def extract_number(text):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1] if nums else None

# 6) Load your modified examples
data = json.load(open(MODIFIED_FILE))
ds   = Dataset.from_list(data)

# 7) Load model + tokenizer + pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda")

# set pad token properly
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    pad_token_id=tokenizer.eos_token_id,
    batch_size=BATCH_SIZE,
)

# 8) Few-shot CoT inference
results = []
for i, item in enumerate(tqdm(ds, desc="Few-shot CoT")):
    prompt = build_few_shot_prompt(item)
    out    = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]["generated_text"]
    pred   = extract_number(out)

    # pull gold from 'gold' (or 'answer' if present)
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

# 9) Save results
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

# 10) Print accuracy
correct = sum(r["correct"] for r in results)
total   = len(results)
print(f" Correct predictions: {correct}/{total}")
