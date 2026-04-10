<img src="Images/cynaptics_logo.jpg" alt="Cynaptics Club Logo" width="150"/>

## GPT-2 Language Model (PyTorch)

A Transformer-based character-level language model built from scratch using PyTorch, inspired by GPT architecture.
The model is trained to predict the next character in a sequence using masked self-attention and autoregressive generation.

**This project demonstrates a complete deep learning pipeline including:**
data preprocessing
token + positional embeddings
multi-head self-attention
feed-forward Transformer blocks
training loop with optimizer + scheduler
text generation with temperature and top-k sampling
Features
Character-level tokenization
Multi-head masked self-attention
6-layer Transformer architecture
Residual connections + LayerNorm
GELU activation
AdamW optimizer
Cosine learning rate scheduler
Gradient clipping
Dropout regularization
Best model checkpoint saving
Temperature + top-k text generation
Model Architecture
Embedding dimension: 256
Attention heads: 8
Transformer layers: 6
Context window: 128
Dropout: 0.2

**The model consists of:**
token embeddings
positional embeddings
stacked Transformer blocks
final linear projection to vocabulary logits
Training

The model is trained on text data using cross-entropy loss for next-character prediction.

**Training optimizations include:**
AdamW
weight decay
cosine annealing LR scheduler
gradient norm clipping
Text Generation

After training, the model generates text autoregressively, one character at a time.

Supports:
temperature scaling
top-k sampling




## A Transformer-based instruction-following language model fine-tuned from pretrained GPT-2 using PyTorch and the Stanford Alpaca dataset.

The model is trained using supervised fine-tuning (SFT) to generate assistant-style responses by predicting the next token in an autoregressive manner.

**This project demonstrates a complete transfer learning pipeline including:**
dataset preprocessing and prompt formatting
BPE tokenization
pretrained GPT-2 loading
manual training loop with CrossEntropyLoss
optimizer-based fine-tuning

**Features**

Instruction-response prompt formatting
BPE tokenization using Hugging Face Transformers
Pretrained GPT-2 base model (124M)
Manual causal language model training loop
Shifted CrossEntropyLoss
AdamW optimizer

**Training**

The model is fine-tuned on the Alpaca instruction dataset using cross-entropy loss for next-token prediction.

Training optimizations include:
AdamW optimizer
weight decay
manual backpropagation
gradient-based weight updates



