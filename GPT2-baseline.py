# Import necessary libraries and modules
import torch
from transformers import (
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate

# Clear any cached memory on the GPU
torch.cuda.empty_cache()

# Determine the device to use (GPU or CPU) based on availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the Reddit dataset and split it into train and test sets
dataset = datasets.load_dataset("reddit")["train"].train_test_split(test_size=0.1)

# Subset the train and test datasets for faster experimentation
train_dataset = dataset["train"][:100000]
test_dataset = dataset["test"][:10000]

# List of columns to remove from the dataset
rem_list = ["author", "body", "id", "normalizedBody", "subreddit", "subreddit_id"]

# Remove unnecessary columns from the train and test datasets
print("\n Removing extra columns.")
[train_dataset.pop(key) for key in rem_list]
[test_dataset.pop(key) for key in rem_list]
print("\nDONE!\n")

# Define the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    gradient_checkpointing=True,
    fp16=True,
    logging_dir="./logs",
    logging_steps=500,
    logging_first_step=True,
    dataloader_num_workers=2,
)


# Define a function to preprocess the data
def preprocess_function(examples):
    inputs = []
    targets = []
    for i in range(len(examples["content"])):
        content = str(examples["content"][i])
        target = str(examples["summary"][i])
        if content.strip() and target.strip():
            inputs.append(content + " TL;DR " + target)
            targets.append(target)

    # Tokenize input sequences
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        return_tensors="pt",
        padding="max_length",
        return_overflowing_tokens=False,
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=32,
            return_tensors="pt",
            padding="max_length",
            return_overflowing_tokens=False,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Convert train and test dictionaries to Dataset objects
train_dataset = datasets.Dataset.from_dict(train_dataset)
test_dataset = datasets.Dataset.from_dict(test_dataset)

# Preprocess the train and test datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Load the ROUGE metric
rouge_score = evaluate.load("rouge")


# Define a function to compute the metrics (ROUGE scores)
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = predictions.argmax(axis=-1)
    # Decode the predicted summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [
        "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores and convert them to percentages
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


# Define a data collator for language modeling tasks
# Set mlm (Masked Language Modeling) to False, as we are using GPT-2 for causal language modeling, not masked language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and evaluate the model
trainer.train()
trainer.evaluate()

# Save the trained model
trainer.save_model("mymodel")

# Alternative saving method and folder
model.save_pretrained("mymodel_alt")
