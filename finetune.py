from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.to(device)

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

def format_alpaca_prompt(example):
    if example["input"].strip():
        text = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"]
        )
    else:
        text = PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"],
            output=example["output"]
        )
    return {"text": text}

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.map(format_alpaca_prompt)

tokenized_dataset = dataset.map(tokenize, batched=True)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

train_loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3

model.train()

for epoch in range(epochs):
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

model.save_pretrained("./fine_tuned_gpt2_alpaca")
tokenizer.save_pretrained("./fine_tuned_gpt2_alpaca")

print("Training complete")