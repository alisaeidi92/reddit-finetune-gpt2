
# Define the device to use (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the Reddit dataset and split it into train and test sets
#dataset = datasets.load_dataset('reddit')['train']

dataset =  datasets.load_dataset('reddit')['train'].train_test_split(test_size=0.1)

train_dataset = dataset['train'][:100000] #keeping 1 million reviews for training
test_dataset = dataset['test'][:10000] # keeping 150k reviews for testing

#train_dataset = datasets.load_dataset('reddit')['train'].shuffle(seed=42)
#test_dataset = datasets.load_dataset('reddit')['train'].shuffle(seed=42)


rem_list = ['author', 'body', 'id', 'normalizedBody', 'subreddit', 'subreddit_id']

print("\n Removing extra columns.")
[train_dataset.pop(key) for key in rem_list]
[test_dataset.pop(key) for key in rem_list]
print("\nDONE!\n")

###print("\n\n\nDataset:\n", dataset)

##dataset = dataset.remove_columns(['author', 'body', 'id', 'normalizedBody', 'subreddit', 'subreddit_id'])

#print("\n\n keys: ", list(dataset.keys()))

##train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

##print("\n\n\nTRAIN DATA KEYS:", train_dataset.keys())
##print("\n\n\nTEST DATA KEYS:", test_dataset.keys())
##print("\n\n\nTRAIN DATA CONTENT: \n", train_dataset['content'][0])
##print("\n\n\nTEST DATA CONTENT: \n", test_dataset['content'][0])
##print("\n\n\nTRAIN DATA SUMM: \n", train_dataset['summary'][0])
##print("\n\n\nTEST DATA SUMM: \n", test_dataset['summary'][0]) 

##exit()

# Define the GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    gradient_checkpointing=True,
    fp16=True,
    logging_dir='./logs',
    #auto_find_batch_size=True,
    logging_steps=500,
    logging_first_step=True,
    dataloader_num_workers=2,
    #predict_with_generate=True,
    #use_cache=False
)

# Define the function to preprocess the data
def preprocess_function(examples):
    ##print("\n\nLEN EXAMPLE:", len(examples))
    ##print("\n\nTYPE EXAMPLE: ", type(examples))
    ##print("\n\n[0] EXAMPLE:", list(examples.keys()))
    ##print("\n\nEXAMPLE: ", examples)
    inputs = []
    targets = []
    for i in range(len(examples['content'])):
        content = str(examples['content'][i])
        target = str(examples['summary'][i])
        if content.strip() and target.strip():
            inputs.append(content + " TL;DR " + target)
            targets.append(target)

            ##print(f"Content: '{content}'")
            ##print(f"Target: '{target}'")
            ##print("\n"*10)
    #
    model_inputs = tokenizer(inputs, max_length=512, return_tensors="pt", padding='max_length', return_overflowing_tokens=False, truncation=True)
    with tokenizer.as_target_tokenizer():
       labels = tokenizer(targets, max_length=32, return_tensors="pt", padding='max_length', return_overflowing_tokens=False, truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# convert train and test dictionaries to Dataset object.
train_dataset = datasets.Dataset.from_dict(train_dataset)
test_dataset = datasets.Dataset.from_dict(test_dataset)

# Preprocess the train and test datasets
train_dataset = train_dataset.map(preprocess_function, batched=True) #remove_columns=["content", "summary"])
test_dataset = test_dataset.map(preprocess_function, batched=True) # remove_columns=["content", "summary"])

#print("\n\nlen train dataset: ", len(train_dataset))
#print("\n\nlen test dataset: ", len(test_dataset))
#print("\n\ntype train dataset: ", type(train_dataset))
#print("\n\ntype test dataset: ", type(test_dataset))


# Define the ROUGE metric
##rouge = datasets.load_metric('rouge')

# Define the compute metrics function
"""
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=['rouge2'])['rouge2'].mid.fmeasure
    return {'rouge2': rouge_output}
"""
  

rouge_score = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    #predictions, labels = eval_pred
    # Decode generated summaries into text
    #print("\n\n PREDICTION: \n", predictions[0])                                                                                                                                                                                            
    #print("\n\n LEN:n", len(predictions))
    #print("\n\n type:", type(predictions))
    #print("\n\n shape:", predictions.shape)
#   pred_ids = predictions['input_ids']
#   labels_ids = labels['input_ids']
    #predictions = predictions.reshape(-1, predictions.shape[-1])  # Reshape to 2D array
    #predictions = predictions.tolist()
    #labels = labels.tolist()

    #predictions = predictions.reshape(-1, )
    #labels = labels.reshape(-1, )
    predictions = predictions.argmax(axis=-1)     
    #predictions = predictions.transpose(2,0,1).reshape(-1,predictions.shape[1])
    #labels = labels.transpose(2,0,1).reshape(-1,labels.shape[1])
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    #print("\n\n\n\n decoded: \n", decoded_preds)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value* 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()

trainer.save_model('mymodel')

# alternative saving method and folder
model.save_pretrained('mymodel_alt')
