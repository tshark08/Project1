Step 1: Import necessary libraries and load the labeled data.
Step 2: Tokenize the text data using a pre-trained tokenizer.
Step 3: Encode the labels into numerical format.
Step 4: Define a custom dataset class to handle tokenized data.
Step 5: Convert tokenized data into dictionary format and create the dataset object.
Step 6: Load a pre-trained bart-large model for sequence classification.
Step 7: Define training arguments.
Step 8: Initialize the Trainer class with the model, training arguments, and dataset.
Step 9: Fine-tune the model using the Trainer class.
Step 10: Save the fine-tuned model and tokenizer for future use.

import pandas as pd
from transformers import AutoTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

# Load labeled data
data = pd.read_csv('labeled_topic_representations.csv')

# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
tokenized_data = data.apply(lambda row: tokenize_function({'text': row['text']}), axis=1)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['label'])
tokenized_data['labels'] = encoded_labels

# Create dataset object
class TopicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

encodings = {key: [example[key] for example in tokenized_data] for key in tokenized_data[0].keys() if key != 'labels'}
dataset = TopicDataset(encodings, encoded_labels)

# Load the model
model = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=len(label_encoder.classes_))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset,               # training dataset
    #eval_dataset=eval_dataset,         # evaluation dataset (if you have one)
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('fine-tuned-bart-model')
tokenizer.save_pretrained('fine-tuned-bart-tokenizer')