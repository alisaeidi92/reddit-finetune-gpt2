import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
from rouge import Rouge
import datasets

#torch.cuda.empty_cache()

# Define the device to use (GPU or CPU)
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda')
# Load the Reddit dataset and split it into train and test sets
#dataset = datasets.load_dataset('reddit')['train']

dataset =  datasets.load_dataset('reddit')['train'].train_test_split(test_size=0.1)

train_dataset = dataset['train'][:50]
eval_dataset = dataset['test'][50:55]



rem_list = ['author', 'body', 'id', 'normalizedBody', 'subreddit', 'subreddit_id']

print("\n Removing extra columns.")
[train_dataset.pop(key) for key in rem_list]
[eval_dataset.pop(key) for key in rem_list]
print("\nDONE!\n")
 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def preprocess_data(example):

    input_text=[]
    target_text=[]

    for i in range(len(example['content'])):
        content = str(example["content"][i])
        summary = str(example["summary"][i])
        if content.strip() and summary.strip():
            input_text.append(content + " TL;DR " + summary)
            target_text.append(summary)

    input_ids = tokenizer(input_text, return_tensors="pt", padding="max_length", return_overflowing_tokens=False, truncation=True, max_length=512)["input_ids"]
    with tokenizer.as_target_tokenizer():
        target_ids = tokenizer(target_text, return_tensors="pt", padding="max_length", return_overflowing_tokens=False,  truncation=True, max_length=64)["input_ids"]
    return {"input_ids": input_ids.squeeze(), "target_ids": target_ids.squeeze()}

train_dataset = datasets.Dataset.from_dict(train_dataset)
eval_dataset = datasets.Dataset.from_dict(eval_dataset)

train_preprocessed = train_dataset.map(preprocess_data, batched=True, batch_size=100, remove_columns=["content", "summary"])
eval_preprocessed = eval_dataset.map(preprocess_data, batched=True, batch_size=100, remove_columns=["content", "summary"])

# Prepare the data loaders
train_dataloader = DataLoader(train_preprocessed, batch_size=1, shuffle=True)
eval_dataloader = DataLoader(eval_preprocessed, batch_size=1, shuffle=True)

# Initialize the GPT-2 model
config = GPT2Config.from_pretrained("gpt2")
config.gradient_checkpointing = True
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to(device)

# Training loop
epochs = 2
coh_steps = 3
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = torch.stack(batch["input_ids"]).to(device)
        target_ids = torch.stack(batch["target_ids"]).to(device)
        #input_ids = batch["input_ids"].to(device)
        #target_ids = batch["target_ids"].to(device)

        # Generate a chain of summaries (coh_steps) as input
        coh_input_ids = input_ids
        for _ in range(coh_steps):
            with torch.no_grad():
                coh_output_ids = model.generate(coh_input_ids, max_length=64, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            coh_input_ids = torch.cat((coh_input_ids, coh_output_ids), dim=1)

        # Train with the final summary conditioned on the chain of summaries
        outputs = model(coh_input_ids, labels=target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Step {step}, Loss: {loss.item()}")


# Evaluation
rouge = Rouge()
rouge_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
num_eval_batches = 0

for batch in eval_dataloader:
    input_ids = torch.stack(batch["input_ids"]).to(device)
    target_ids = torch.stack(batch["target_ids"]).to(device)
    #input_ids = batch["input_ids"].to(device)
    #target_ids = batch["target_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=64, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    # Decode the generated summaries and reference summaries
    generated_summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    reference_summaries = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

    # Calculate rouge scores for the batch
    for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
        batch_rouge_scores = rouge.get_scores(gen_summary, ref_summary, avg=True)

        rouge_scores["rouge-1"] += batch_rouge_scores["rouge-1"]["f"]
        rouge_scores["rouge-2"] += batch_rouge_scores["rouge-2"]["f"]
        rouge_scores["rouge-l"] += batch_rouge_scores["rouge-l"]["f"]

    num_eval_batches += 1

# Calculate the average rouge scores
rouge_scores["rouge-1"] /= num_eval_batches
rouge_scores["rouge-2"] /= num_eval_batches
rouge_scores["rouge-l"] /= num_eval_batches

print("Average ROUGE scores:")
print(f"ROUGE-1: {rouge_scores['rouge-1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge-2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rouge-l']:.4f}")
