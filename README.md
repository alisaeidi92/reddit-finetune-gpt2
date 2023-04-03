## 1. Introduction

This project aims to complete the Chain of Hindsight Challenge, which involves fine-tuning a small GPT-2 model with 124 million parameters on a text summarization task using the HuggingFace Reddit dataset. The goal is to leverage the method proposed in the Chain of Hindsight (CoH) paper to improve the model's performance and measure its ROUGE score. The challenge allows the use of any environment and resources, including cloud services (GCP, AWS) or local machines (Bare Metal or Docker).

## 2. Research and Study

During the research phase, an in-depth study of the GPT-2 model, the HuggingFace Reddit dataset, and the Chain of Hindsight (CoH) paper was conducted.

GPT-2 model: The GPT-2 model is a powerful and highly versatile language model developed by OpenAI. It's designed for a variety of natural language processing tasks, including text summarization.
HuggingFace Reddit dataset: This dataset contains over 3 million data points from the Reddit platform, providing long excerpts that can be reduced to summarized strings. The dataset is available at https://huggingface.co/datasets/reddit.
Chain of Hindsight (CoH) paper: The CoH paper proposes a method for finetuning language models on text summarization tasks by leveraging a unique training process that focuses on improving the model's performance iteratively. The paper is available at https://arxiv.org/pdf/2302.02676.pdf.

## 3. Baseline Implementation

A baseline GPT-2 model was fine-tuned on the HuggingFace Reddit dataset for the text summarization task. The following steps were implemented:

1. **Data preprocessing**:
   - Loaded the HuggingFace Reddit dataset and split it into training and validation sets.
   - Tokenized the dataset using the GPT-2 tokenizer, ensuring that input sequences were within the model's maximum sequence length.
   - Created a custom dataset processing function to handle the tokenized data, including padding, truncation, concatenating content and summary and " TD;LR " for GPT2 text summarization task.

2. **Model fine-tuning**:
   - Initialized a GPT-2 model with the HuggingFace Transformers library.
   - Set up an optimizer, learning rate scheduler, and other necessary hyperparameters for the training process.
   - Fine-tuned the GPT-2 model on the preprocessed dataset using a suitable loss function (e.g., CrossEntropyLoss) and trained it for a specified number of epochs.
   - Saved the fine-tuned model for evaluation and future use.

3. **Evaluation**:
   - Due to the limited computational resources and long training time, a comprehensive evaluation using the ROUGE metric could not be completed for the baseline model.

## 4. Implementation of CoH using GPT-2 Baseline

The baseline GPT-2 model was modified to leverage the CoH methods to improve its performance on the Reddit dataset. The following steps were implemented:

1. **Implementing the CoH algorithm**:
   - Studied the CoH paper and understood the iterative process it proposed for fine-tuning language models.
   - Implemented the CoH method, including the necessary functions for handling model training and evaluation.

2. **Fine-tuning the GPT-2 model with CoH**:
   - Integrated the CoH method into the training process of the GPT-2 model.
   - Fine-tuned the GPT-2 model using the CoH method, following the same training steps as in the baseline implementation but with the added iterative process for improving the model's performance.

3. **Evaluation**:
   - Due to the limited computational resources and long training time, a comprehensive evaluation using the ROUGE metric could not be completed for the CoH-modified model.


## 5. Challenges

During the project, several challenges were encountered:

a) Lack of enough computation speed: Training on the full 3M data points took over 220 hours on a powerful machine on Google Cloud (16 core, 104 RAM, and 1 T4 GPU), preventing the models from fully training.

b) GPT-2 baseline was able to produce some ROUGE results on an extremely small dataset due to hardware issues.

c) The CoH GitHub project received a major update during the project, causing unexpected adjustments and changes.

## 6. Benefits

Despite the challenges, the project provided valuable learning experiences and insights:

a) Understanding the flow of a language model using a relatively new algorithm such as GPT.

b) The project served as an educational experience, introducing various new APIs and helpful functions for processing text data and fine-tuning language models.

c) Practical experience in using new metrics, such as ROUGE, for evaluating the performance of the model on the text summarization task.

d) The opportunity to discover the similarities and differences between deep learning techniques used in computer vision and those employed in natural language processing, particularly in terms of familiar functions and optimizers. This helped to better understand the versatility of deep learning techniques across different domains.

## 7. Techniques used to overcome memory errors:

   CUDA (Out of Memory) OOM runtime error. Used various methods to overcome the OOM errors:
   
   i) Decreasing batch size and Enabling 'auto_find_batch_size' (lead to 0 batch size during the run)
   
   ii) Randomly sampling the dataset and reducing the memory usage
   
   iii) Enabled gradient checkpointing which saves a lot of memory during training
   
   iv) Using no_grad context-manager to detach gradients for evaluation and saving memory
   
   v) Improving system performance by increasing the number of cores and RAM size
