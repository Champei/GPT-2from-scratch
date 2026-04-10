import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model

model_path = "./fine_tuned_gpt2_alpaca"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

def generate_response(instruction, input_text=""):
    if input_text.strip():
        prompt = PROMPT_WITH_INPUT.format(
            instruction=instruction,
            input=input_text
        )
    else:
        prompt = PROMPT_WITHOUT_INPUT.format(
            instruction=instruction
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return generated_text

print(generate_response("Explain machine learning in simple words"))
print("\n")
print(generate_response("Write a short poem about rain"))