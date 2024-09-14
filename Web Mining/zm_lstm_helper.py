## Imports and constants
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(20) 
np.random.seed(20)


MAX_SEQ_LEN = 200
BATCH_SIZE = 10


################ Dataset class ################
    ## Uses preprocessing method above
    ## __getitem__ returns (preprocessed) text and its corresponding label

##TODO: Augment such that it can return both text and vector forms of reviews
class ReviewsDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        ret_review = self.reviews[idx]
        ret_label = self.labels[idx]
        
        return ret_review, ret_label
    

################ Padding function ################
def pad_review(review: torch.Tensor, embedding_dim, max_seq_len=MAX_SEQ_LEN):

    padded_review = review
    pad_tensor = torch.zeros(embedding_dim) ## Pad with zero tensor of size equal to word embeddings

    if len(review) < max_seq_len:
            padded_review = padded_review + [pad_tensor for i in range(max_seq_len - len(review))]
    elif len(review) > max_seq_len:
        padded_review = padded_review[:max_seq_len]

    return padded_review


################ Collate function ################
## Function to pad or trim reviews to same number of tokens
def review_collate_fn(raw_batch):
    ## Input: Collection of (review, label) tuples from ReviewDataset

    padded_reviews = []
    labels = []

    for (review, label) in raw_batch:
        padded_review = pad_review(review, len(raw_batch[0][0][0]))
        padded_reviews.append(padded_review)
        labels.append(label)
    
    # print(torch.Tensor(padded_reviews).shape)

    ## Returns: a tuple (review tensor, label tensor) of sizes batch_size*MAX_SEQ_LEN and batch_size, respectively.
    return torch.Tensor(padded_reviews), torch.Tensor(labels)



################ Evaluation function ################
def evaluation(model:nn.Module, data_loader:DataLoader, loss_fn=nn.CrossEntropyLoss(), device=DEVICE):
    with torch.no_grad():
        model.eval()

        total_correct = 0
        total_loss = 0
        all_pred = []
        all_thought_vectors = []
        all_labels = []

        ## Process batch by batch
        for reviews, labels in data_loader:
            reviews, labels = reviews.to(device), labels.to(device)

            ## Forward pass
            pred, thought_vector = model(reviews)
            pred_class = torch.argmax(pred, dim=1).cpu() ## Get indices of largest value in prediction tensor, ie predicted class

            ## Calculate loss
            loss = loss_fn(pred, labels.long())
            total_loss += abs(loss)

            pred, thought_vector = pred.to('cpu'), thought_vector.to('cpu')
            all_pred.extend(pred_class.numpy())
            all_thought_vectors.extend(thought_vector.numpy())
            all_labels.extend(labels.to('cpu').numpy())
        
        ## Calculate metrics
        correct_preds = sum([1 for pred, label in zip(all_pred, all_labels) if pred == label])
        f1 = f1_score(all_labels, all_pred, average='weighted')

    return total_loss, f1



################ Class to handle early stopping ################
class EarlyStopper():
    def __init__(self, patience=2, delta=0.05, path='best_model.pt'):
        self.patience = patience
        self.counter = 0
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.path = path

    
    ## Checks if early stopping is necessary
    @torch.no_grad()
    def __call__(self, model, val_loss):
        model.eval()

        if self.best_loss == None: 
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            print(f'Initialize baseline model, loss: {val_loss}')
        ## If val loss deteriorates by a certain degree, exhaust patience
        elif val_loss >= self.best_loss + self.delta:
            self.counter += 1
            print(f'val_loss deteriorated, count: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        ## If val loss improves, save model and restart counter
        else: 
            self.best_loss > val_loss
            torch.save(model.state_dict(), self.path)
            print(f'New best model saved, loss: {val_loss}')
            self.counter = 0



################ Training function ################
def train_model(
    model:nn.Module, 
    train_loader:DataLoader, 
    val_loader:DataLoader, 
    lr=0.01, 
    epochs=3, 
    loss_fn=nn.CrossEntropyLoss(),
    early_stopper:EarlyStopper=None,
    device=DEVICE
):
    model = model.to(device)

    ## Initialize optimizer for params that require gradients
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)

    ## Training loop
    for epoch in range(epochs):
        train_loss = 0
        epoch_all_pred = []
        epoch_all_labels = []

        ## Iterate through batches
        for reviews, labels in train_loader:
            model.train()
            reviews, labels = reviews.to(device), labels.to(device)

            ## Forward pass
            pred, _ = model(reviews)
            pred_class = torch.argmax(pred, dim=1) ## Get indices of largest value in prediction tensor, ie predicted class

            ## Calculate loss
            loss = loss_fn(pred, labels.long())
            train_loss += abs(loss)

            ## Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## Tabulate epoch outputs
            epoch_all_pred.extend(pred_class.cpu().numpy())
            epoch_all_labels.extend(labels.cpu().numpy())

        ## Epoch statistics
        train_f1 = f1_score(epoch_all_labels, epoch_all_pred, average='weighted')
        # if epoch == 0 or (epoch + 1) % 100 == 0:
        print(f'======== Epoch {epoch + 1} ========')
        print(f'Training loss: {train_loss / len(train_loader):.4f}\t', f'Training F1: {train_f1}')

        ## Validation
        val_loss, val_f1 = evaluation(model, val_loader, loss_fn=loss_fn, device=device)
        print(f'Validation loss: {val_loss / len(val_loader):.4f}\t', f'Validation F1: {val_f1}')

        ## If necessary, check early stopping criteria
        if early_stopper != None:
            early_stopper(model, val_loss / len(val_loader))
            if early_stopper.early_stop:
                break

    return model



################ Calculate F1 score for each class ################
def scoring(y_true, y_pred, labels:list=[0,1,2]):

    f1_dict = {label: 0 for label in labels}

    for label in labels:
        ## First convert multi-class lists to two classes
        curr_y_pred = [1 if label == p else 0 for p in y_pred]
        curr_y_true = [1 if label == p else 0 for p in y_true]

        f1_dict[label] = f1_score(curr_y_true, curr_y_pred)

    return f1_dict

    # cm = confusion_matrix(actual_list, prediction_list)
    # plt.figure(figsize=(6, 6))
    # sns.heatmap(cm, annot=True, cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix of Sentiment (Count)')
    # plt.show()   
        
    # plt.figure(figsize=(6, 6))
    # sns.heatmap(cm/np.sum(cm), annot=True, cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix of Sentiment (Percentage)')
    # plt.show()

    return