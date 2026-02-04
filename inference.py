from transformers import AutoTokenizer
from trl import AutoModelForRewardModel
import torch

# Load your trained reward model
model_path = "./Reward-Model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForRewardModel.from_pretrained(model_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_reward(prompt, completion):
    """
    Returns the reward score for a given prompt + completion pair.
    """
    text = prompt + completion
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        reward = model(input_ids=input_ids, attention_mask=attention_mask).logits

    return reward.item()


if __name__ == "__main__":
    # Example usage
    prompt = "Explain the benefits of exercise."
    completion = "Exercise improves mental and physical health by increasing energy and focus."

    score = get_reward(prompt, completion)
    print(f"Reward score: {score:.4f}")

    # You can also score multiple completions
    completions = [
        "Exercise is good for health.",
        "Sitting all day is the best for productivity."
    ]
    for c in completions:
        print(f"Completion: {c}\nScore: {get_reward(prompt, c):.4f}\n")
