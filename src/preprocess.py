from datasets import load_dataset
from transformers import AutoTokenizer

def tokenization(data):
    return tokenizer(data["content"])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("elricwan/HarryPotter", split="train")
dataset = dataset.map(tokenization, batched=True)

print(dataset)