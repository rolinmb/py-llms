import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(tokenizer, prompt, device, model):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs.input_ids, max_length=100)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    prompt = 'What is an llm?'
    generated_text = generate_text(tokenizer, prompt, device, model)
    print("\n"+generated_text)
