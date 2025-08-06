import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Verify SentencePiece
import sentencepiece

print("SentencePiece version:", sentencepiece.__version__)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Load additional datasets with adjusted paths
try:
    with open("data/skin_issues.json", "r", encoding="utf-8") as f:
        skin_issues = {issue["name"]: issue for issue in json.load(f)["skin_issues"]}
except FileNotFoundError:
    print("Warning: data/skin_issues.json not found. Using only quick_tune.jsonl.")
    skin_issues = {}
try:
    with open("data/compatibility_rules.json", "r", encoding="utf-8") as f:
        compatibility_rules = json.load(f)
        conflicts = {c["ingredient_A"]: c["ingredient_B"] for c in compatibility_rules["conflicts"]}
except FileNotFoundError:
    print("Warning: data/compatibility_rules.json not found. No conflict data.")
    conflicts = {}


# Dynamically generate training data
def generate_training_examples():
    examples = []
    # Load existing quick_tune.jsonl
    try:
        with open("AI/quick_tune.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                examples.append({"input": example["input"], "output": example["output"]})
    except FileNotFoundError:
        print("Warning: AI/quick_tune.jsonl not found, using only dynamic examples.")

    # Generate examples from skin_issues if available
    if skin_issues:
        for issue in skin_issues.values():
            skin_type = issue.get("skin_type", "unknown")
            if skin_type in ["oily", "dry", "sensitive", "combination"]:
                ingredient = issue.get("suggested_ingredient", "aloe vera")
                safety_tip = f"Steer clear of {conflicts.get(ingredient, 'no conflicts')}" if conflicts.get(
                    ingredient) else "No specific conflicts"
                prompt = f"Hey! Create a fun {skin_type} skin mask for {issue['name']} using {ingredient}. Tackle: {issue['name']}. Give me 3 chill steps: Prepare, Apply, Remove. Safety tip: {safety_tip}."
                output = f"Hey Yaar! Here’s a {skin_type} skin mask for {issue['name']} with {ingredient}. 1. Prepare: Mix 1 tsp {ingredient} with 2 tbsp water. 2. Apply: Spread gently for 10 mins. 3. Remove: Rinse with cool water. Pro tip: {safety_tip}!"
                examples.append({"input": prompt, "output": output})
    return examples


# Load and tokenize dataset
dataset = generate_training_examples()
print("Dataset sample:", dataset[:5])

dataset = Dataset.from_dict({"input": [ex["input"] for ex in dataset], "output": [ex["output"] for ex in dataset]})


def tokenize_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["input", "output"]
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/fine_tuned_model",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=50,
    logging_dir="./logs",
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

# Train and save
trainer.train()
model.save_pretrained("./models/fine_tuned_model")
tokenizer.save_pretrained("./models/fine_tuned_model")