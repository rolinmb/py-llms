import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

if __name__ == "__main__":
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator
    )

    trainer.train()

    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        outputs = model.generate(inputs.input_ids, max_length=100)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    prompt = 'What is an llm?'
    generated_text = generate_text(prompt)
    print("\n"+generated_text)