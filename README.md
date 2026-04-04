<img src="Images/cynaptics_logo.jpg" alt="Cynaptics Club Logo" width="150"/>

# Cynaptics Induction Tasks March 2026

We are the Cynaptics Club of IIT Indore, the AI/ML enthusiasts' hub, and we're excited to present you with our induction task!

Submission guidelines and detailed explanations for each task can be found in their respective task folders. Best of luck, and we look forward to your innovative solutions!


# Task 1

## Make your own GPT: The "Glorified Autocomplete" (GPT-2 Pretraining)

Build a scaled-down, **decoder-only Transformer** (GPT-2 style) from scratch using **PyTorch** and train it on the [Tiny Shakespeare] dataset. The goal is to autocomplete the text — essentially a highly complex autocomplete.**The model architecture should be based on GPT-2 but you are free modify it.**

### Core Deliverables
1. **Data Loader** — Tokenize the dataset into input/target pairs (shifted by one token).
2. **Transformer Architecture** — Masked Self-Attention, Multi-Head Attention, Feed-Forward Networks, Positional & Token Embeddings.
3. **Pretraining Loop** — Train with Cross-Entropy Loss over multiple epochs.
4. **Text Generation** — Sample from the model's probability distribution to generate Shakespeare-esque text.

> **Note:** You have to attempt Task-1 before moving to Task 2.

Full details, milestones, and submission guidelines are in the [Task 1 folder](./Task1/).

# Task 2 (Optional)

## Fine-Tune GPT-2: The "Instruction Follower" (Supervised Fine-Tuning)

Take the pretrained **[GPT-2 base (124M)](https://huggingface.co/openai-community/gpt2)** model from Hugging Face and fine-tune it on the **[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)** instruction-following dataset. The goal is to transform a generic language model into an assistant that can follow instructions and generate helpful responses.

### Core Deliverables
1. **Dataset Preparation** — Load and format the Alpaca dataset using the standard prompt template.
2. **Model Loading** — Load the pretrained GPT-2 model and tokenizer from Hugging Face.
3. **Fine-Tuning Loop** — Train (SFT) the model with Cross-Entropy Loss on the formatted instruction data.
4. **Inference** — Generate assistant-style responses from new instruction prompts.

Full details, milestones, and submission guidelines are in the [Task 2 folder](./Task2/).


# Deadlines
Deadline for submission of both Task 1 and Task 2 will be **10th April EOD**, via a google form that will be circulated soon.
