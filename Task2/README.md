***

# Task 2: Supervised Fine-Tuning of GPT-2 on Alpaca (Optional Task)

**Fine-Tuning**. Your objective is to take a pretrained GPT-2 base model and fine-tune it on the Stanford Alpaca dataset to follow instructions in a conversational assistant style.

Unlike Task 1 where you built and pretrained a model from scratch, here you will leverage the power of **transfer learning** — starting from a model that already understands language and adapting it to a new downstream task.

## 📋 Task 2 Overview

You will load the pretrained **GPT-2 base (124M)** model from Hugging Face and perform **Supervised Fine-Tuning (SFT)** on the Alpaca instruction-following dataset. By the end, your model should be able to take an instruction (and optional input) and generate a helpful assistant-style response.

### Pretrained Model
* **Hugging Face Link:** [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)

### Dataset
* **Alpaca Dataset:** [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

The Alpaca dataset contains ~52K instruction-following examples, each with:
- `instruction` — The task description
- `input` — Optional additional context  
- `output` — The expected assistant response

### Core Deliverables:

**All deliverable code should be python files, other formats like jupyter notebooks are not allowed**.

1. **Dataset Preparation:** Load and format the Alpaca dataset into a prompt template suitable for causal language model fine-tuning.
2. **Model Loading:** Load the pretrained GPT-2 base model and tokenizer from Hugging Face.
3. **Fine-Tuning Loop:** Implement a training loop to fine-tune the model with Cross-Entropy Loss (or any other loss).
4. **Inference:** A generation script that takes an instruction prompt and generates an assistant-style response.

## 💬 Prompt Template

Use the following prompt format to structure each training example (given in alpaca):

**With input context:**
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**Without input context:**
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
```

## 🗺️ Suggested Milestones

* **Step 1: Load & Explore the Dataset**
    * Use the provided `DataLoader.py` or write your own to load and format the Alpaca dataset.
* **Step 2: Load the Pretrained Model**
    * Load `openai-community/gpt2` from Hugging Face along with its tokenizer. Set the pad token appropriately.
* **Step 3: Format & Tokenize**
    * Apply the prompt template to each example and tokenize for causal LM training (input = target, shifted internally).
* **Step 4: Fine-Tune**
    * Train the model for a few epochs. Monitor training loss and optionally evaluate on a held-out split.
* **Step 5: Generate & Evaluate**
    * Write an inference script that formats a new instruction using the prompt template and generates a response.

## 📤 Submission Guidelines

1. Develop your solution in the `Task2` folder of your forked repo.
2. Ensure your code is clean, modular, and easy to follow.
3. Include a **README** with:
    * Instructions on how to run your fine-tuning script.
    * Instructions on how to run your inference/generation script.
    * Sample outputs from your fine-tuned model (at least 3 different instructions).
    * Hyperparameters used (learning rate, batch size, epochs, etc.).
    * Training loss curve or final loss value.
4. Submit via the induction submission form.

## ⚖️ Evaluation Criteria

- **Understanding of SFT**: Do you understand the difference between pretraining and fine-tuning?
- **Data Formatting**: Is the dataset properly formatted with the prompt template?
- **Code Quality**: Is the code organized, documented, and hand written?
- **Results Quality**: Does the fine-tuned model produce coherent, instruction-following responses?
- **Conceptual Grasp**: Can you explain the fine-tuning process and design choices during your induction interview?

## 📚 Helpful Resources
* [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
* [GPT-2 Model Card](https://huggingface.co/openai-community/gpt2)
* [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
***
