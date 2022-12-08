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
from torch.utils.data import RandomSampler, TensorDataset, DataLoader

# Training and validation set dataLoader from pytorch

batch_size = 64
train_set_data = TensorDataset(train_tensor, train_attention_mask, train_labels)
val_set_data = TensorDataset(val_tensor, val_attention_mask, val_labels)
train_set_dataloader = DataLoader(train_set_data, sampler=RandomSampler(train_set_data), batch_size=batch_size)
val_set_dataloader = DataLoader(val_set_data, sampler=RandomSampler(val_set_data), batch_size=batch_size)

# freeze all the parameters for fine-tuning 
for parameter in bert.parameters():
    parameter.requires_grad = False
    
# Initializes BERT architecture

fc1_input_size = 768
fc1_output_size = 512
fc2_output_size = 48
fc3_output_size = 2

class BERT(nn.Module):

    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(fc1_input_size, fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)
        self.fc3 = nn.Linear(fc2_output_size, fc3_output_size)
        self.dropout = nn.Dropout(0.08)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sent_id, mask):
        _, input = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
        model = self.fc1(input)
        model = self.fc2(model)
        model = self.fc3(model)
        model = self.dropout(model)
        model = self.leakyrelu(model)
        model = self.softmax(model)

        return model
    
model = BERT(bert)
model = model.to(device)

# Computes weights and defines hyperparameters (optimizer, loss function, number of epochs)

weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_set_labels), y = train_set_labels), dtype = torch.float)
weights = weights.to(device)
binary_cross_entropy = nn.NLLLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(),lr = 0.00001) 
num_epochs = 20

# Function to compute loss and train the model

def train():
    
    model.train()
    preds = []
    loss = 0
  
    for i, batch in enumerate(train_set_dataloader):
            
        batch = [b.to(device) for b in batch]
        sent_id, mask, labels = batch
        model.zero_grad()        
        curr_preds = model(sent_id, mask)
        curr_loss = binary_cross_entropy(curr_preds, labels)
        loss += curr_loss.item()
        curr_loss.backward()
        optimizer.step()
        curr_preds = curr_preds.detach().cpu().numpy()

    loss_average = loss / len(train_set_dataloader)
    preds.append(curr_preds)
    preds = np.concatenate(preds, axis = 0)

    return loss_average, preds
  
# Function to evaluate the model on the validation set

def evaluate():

    model.eval()
    preds = []
    accuracy = 0
    loss = 0

    for i, batch in enumerate(val_set_dataloader):
        batch = [b.to(device) for b in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            curr_preds = model(sent_id, mask)
            curr_loss = binary_cross_entropy(curr_preds, labels)
            loss += lcurr_loss.item()
            curr_preds = curr_preds.detach().cpu().numpy()
            preds.append(curr_preds)
 
    preds = np.concatenate(preds, axis = 0)
    loss_average = loss / len(val_set_dataloader)

    return loss_average, preds
  
# Iterate through training and validation with num_epochs and print loss

min_loss = float('inf')

train_set_loss_list = []
val_set_loss_list = []

for i in range(num_epochs):
    print('\n Epoch :' + str(i + 1))
    train_set_loss, _ = train()
    val_set_loss, _ = evaluate()
    
    print('\nTraining Loss: ' + str(train_set_loss))
    print('\nValidation Loss: ' + str(val_set_loss))

    if val_set_loss < min_loss:
        min_loss = val_set_loss

    train_set_loss_list.append(train_set_loss)
    val_set_loss_list.append(val_set_loss)
    
# Evaluates model performance on test set

with torch.no_grad():
    test_predictions = model(test_torch.to(device), test_attention_mask.to(device))
    test_predictions = test_predictions.detach().cpu().numpy()
    
# Print the model performance (Accuracy, F1-Score, Precision, Recall, etc.)

test_predictions = np.argmax(test_predictions, axis = 1)
print(classification_report(test_labels, test_predictions))
