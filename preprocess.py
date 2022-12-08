import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset, DataLoader

device = torch.device("cuda")
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
df = pd.read_csv("combined_data_v3.csv")

# removes irrevelant columns from the original data
df = df.drop(['patient_id', 'uniq_id', 'chart_labeled_date', 'Unnamed: 0'], axis = 1)

# replaces True and Falses with 1 and 0 to make outcome more straightforward
df = df.replace(True, 1)
df = df.replace(False, 0)

# Using train test split, we split the text into training, validation, and test sets

train_set = train_test_split(df['text'], df['outcome'], random_state = 2022, test_size = 0.8, stratify = df['outcome'])[0]
val_test_set = train_test_split(df['text'], df['outcome'], random_state = 2022, test_size = 0.8, stratify = df['outcome'])[1]
train_set_labels = train_test_split(df['text'], df['outcome'], random_state = 2022, test_size = 0.8, stratify = df['outcome'])[2]
val_test_set_labels = train_test_split(df['text'], df['outcome'], random_state = 2022, test_size = 0.8, stratify = df['outcome'])[3]

val_set = train_test_split(val_test_set, val_test_set_labels, random_state = 2022, test_size = 0.5, stratify = val_test_set_labels)[0]
test_set = train_test_split(val_test_set, val_test_set_labels, random_state = 2022, test_size = 0.5, stratify = val_test_set_labels)[1]
val_set_labels = train_test_split(val_test_set, val_test_set_labels, random_state = 2022, test_size = 0.5, stratify = val_test_set_labels)[2]
test_set_labels = train_test_split(val_test_set, val_test_set_labels, random_state = 2022, test_size = 0.5, stratify = val_test_set_labels)[3]

for i in range(15):
    ind = train_set.index[i]
    train_set[ind] = train_set[ind] + " Q: Does this person have food insecurity? A: " + str(train_set_labels[ind])

train_set = train_set.tolist()
val_set = val_set.tolist()
test_set = test_set.tolist()
train_set_labels = train_set_labels.tolist()
val_set_labels = val_set_labels.tolist()
test_set_labels = test_set_labels.tolist()

# Converts list of labels to tensor

train_labels = torch.tensor(train_set_labels)
val_labels = torch.tensor(val_set_labels)
test_labels = torch.tensor(test_set_labels)

# Tokenizes and encodes sequences in the training, validation, and test sets

max_sentence = 100

tokens_train = tokenizer.batch_encode_plus(train_set, max_length = max_sentence, padding='max_length', truncation=True)
tokens_val = tokenizer.batch_encode_plus(val_set, max_length = max_sentence, padding='max_length', truncation=True)
tokens_test = tokenizer.batch_encode_plus(test_set, max_length = max_sentence, padding='max_length', truncation=True)

# Applies attention masks and converts tokenized lists to tensors

train_attention_mask = torch.tensor(tokens_train['attention_mask'])
val_attention_mask = torch.tensor(tokens_val['attention_mask'])
test_attention_mask = torch.tensor(tokens_test['attention_mask'])
train_tensor = torch.tensor(tokens_train['input_ids'])
val_tensor = torch.tensor(tokens_val['input_ids'])
test_tensor = torch.tensor(tokens_test['input_ids'])
