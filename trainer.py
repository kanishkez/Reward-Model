from transformers import AutoTokenizer
from datasets import load_dataset
from trl import RewardTrainer, TRLTrainingArguments, AutoModelForRewardModel

model_name = "qwen-2.5-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForRewardModel.from_pretrained(model_name)

# Load the Anthropic RLHF dataset
dataset = load_dataset("Anthropic/rlhf")["train"]

def tokenize_fn(batch):
    chosen = [p + r for p, r in zip(batch["prompt"], batch["chosen"])]
    rejected = [p + r for p, r in zip(batch["prompt"], batch["rejected"])]
    chosen_enc = tokenizer(chosen, truncation=True, padding=True)
    rejected_enc = tokenizer(rejected, truncation=True, padding=True)
    return {
        "input_ids_chosen": chosen_enc["input_ids"],
        "attention_mask_chosen": chosen_enc["attention_mask"],
        "input_ids_rejected": rejected_enc["input_ids"],
        "attention_mask_rejected": rejected_enc["attention_mask"],
    }
# Tokenize the data for RewardTrainer
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

training_args = TRLTrainingArguments(
    output_dir="./Reward-Model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    bf16=True,
    optim="adamw_torch",
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
#Train
trainer.train()
