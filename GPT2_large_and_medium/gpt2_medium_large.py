# this file was used to run the gpt models on the test dataset of GSM8K for zero shot and few shot COT. 

# import necessary libraies
import re
import json
import torch
import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    pipeline,
)
from tqdm.auto import tqdm

# 1. Configuration - adjust to change model and other parameters for fine tuning
model_names = ["gpt2-medium", "gpt2-large"]
output_dir = "models"
batch_size = 8            # increase batch size for full dataset
num_train_epochs = 3      # full training epochs
learning_rate = 5e-5

# 2. Subset sizes (None means use the full dataset)
subset_train_size = None
subset_test_size = None

# 3. Few-shot examples with chain-of-thought
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

# Helper: extract last number from generated text
def extract_number(text):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1] if nums else None

# Load GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")
train_data = gsm8k["train"]
test_data = gsm8k["test"]

#  select subsets for quick testing
if subset_train_size is not None:
    train_data = train_data.select(range(subset_train_size))
if subset_test_size is not None:
    test_data = test_data.select(range(subset_test_size))

# Build few-shot prompt function
def build_few_shot_prompt(item):
    prompt = ""
    for ex in few_shot_examples:
        prompt += f"{ex['question']} {ex['answer']}\n\n"
    prompt += f"Q: {item['question']} Let's think step by step:"
    return prompt

# Ensure output directory
os.makedirs(output_dir, exist_ok=True)

for base_ckpt in model_names:
    model_id = base_ckpt.split('/')[-1]
    print(f"\n=== Processing {model_id} ===")

    # --- Pre-FT inference ---
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(base_ckpt).to("cuda")
    tokenizer.pad_token = tokenizer.eos_token
    pre_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        device=0, pad_token_id=tokenizer.eos_token_id, batch_size=4
    )

    pre_correct = []
    for i, item in enumerate(tqdm(test_data, desc="Pre-FT Inference")):
        prompt = item['question'] + " Let's think step by step:"
        out = pre_pipe(prompt, max_new_tokens=100, do_sample=False)
        pred = extract_number(out[0]['generated_text'])
        gold = extract_number(item['answer'])
        if pred == gold:
            pre_correct.append({'id': i, 'question': item['question'], 'gold': gold, 'pred': pred})
    with open(os.path.join(output_dir, f"pre_ft_correct_{model_id}.json"), "w") as f:
        json.dump(pre_correct, f, indent=2)

    # --- Fine-tuning ---
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(base_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        tok = tokenizer(examples['question'], examples['answer'],
                        truncation=True, padding='max_length', max_length=512)
        tok['labels'] = tok['input_ids'].copy()
        return tok

    tokenized_train = train_data.map(tokenize_fn, batched=True)
    tokenized_train.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{model_id}-ft"),
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_total_limit=1,
        logging_steps=200,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_train)
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, f"{model_id}-ft"))
    tokenizer.save_pretrained(os.path.join(output_dir, f"{model_id}-ft"))

    # --- Post-FT inference on full set ---
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, f"{model_id}-ft"))
    model = AutoModelForCausalLM.from_pretrained(os.path.join(output_dir, f"{model_id}-ft")).to("cuda")
    ft_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        device=0, pad_token_id=tokenizer.eos_token_id, batch_size=4
    )

    for mode in ['zero', 'few']:
        results, correct = [], 0
        desc = f"Post-FT {mode}-shot"
        for i, item in enumerate(tqdm(test_data, desc=desc)):
            prompt = (item['question'] + " Let's think step by step:") if mode=='zero' else build_few_shot_prompt(item)
            out = ft_pipe(prompt, max_new_tokens=100, do_sample=False)
            pred = extract_number(out[0]['generated_text'])
            gold = extract_number(item['answer'])
            is_corr = pred == gold
            results.append({'id': i, 'question': item['question'], 'gold': gold, 'pred': pred, 'correct': is_corr})
            if is_corr:
                correct += 1
        acc = correct / len(test_data)
        print(f"{desc} accuracy for {model_id}: {acc:.2%}")
        with open(os.path.join(output_dir, f"{mode}_shot_{model_id}.json"), "w") as f:
            json.dump(results, f, indent=2)

print("All steps complete.")