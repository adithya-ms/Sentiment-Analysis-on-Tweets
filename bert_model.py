import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from baseline_models import splitData, print_confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer
import preprocessing
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description="Sentiment analysis")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)


class Dataset(Dataset):
    def __init__(self, data, labels_encoded, tokenizer):
        super().__init__()
        # Making lists for: labels, input_ids, attention_masks
        labels = []
        input_ids = []
        attention_masks = []
        for line, label in zip(data, labels_encoded):
            encoded_dict = tokenizer.encode_plus(
                line,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]' tokens to sentences
                max_length=64,  # Pad & truncate all sentences to max_length.
                pad_to_max_length=True,
                return_attention_mask=True,  # Return attention masks
                return_tensors='pt',  # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(label)

        self.input_ids = torch.cat(input_ids)
        self.attention_masks = torch.cat(attention_masks)
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.input_ids[index].long().to(device), self.attention_masks[index].long().to(device), self.labels[
            index].long().to(device)

    def __len__(self):
        return len(self.input_ids)

#Evaluating the model
def evaluate(model, loader, le):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        y_true = []
        y_pred = []

        for batch in loader:
            input_ids, attention_masks, labels = batch
            out = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels,
                        return_dict=True)
            preds = out.logits.argmax(dim=-1)

            output = out.logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(output)
            labels_ = labels.cpu().numpy()
            y_true.extend(labels_)

            correct += (preds == labels).sum().item()
            total += labels.numel()

    print("predicited ", correct, " out of ", total, " Correctly")
    model.train()
    return correct / total, y_true, y_pred

#Training the model
def train(model, train_loader, valid_loader, epochs, le):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0.0
        for i, batch in enumerate(train_loader):

            if i % 5 == 0 and i > 0:
                tps = total_tokens / (time.time() - start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, toks/sec: %d" % (epoch, i, total_loss / 5, tps))
                start_time = time.time()
                total_loss = 0.0
                total_tokens = 0.0

            input_ids, attention_masks, labels = batch
            optimizer.zero_grad()
            out = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels,
                        return_dict=True)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
            total_tokens += input_ids.numel()

        acc, y_true, y_pred = evaluate(model, valid_loader, le)
        print("[Epoch %d] Acc (valid): %.4f" % (epoch, acc))
        start_time = time.time()

    print("############## END OF TRAINING ##############")
    acc, y_true, y_pred = evaluate(model, valid_loader, le)
    print("Confusion matrix BERT:")
    y_pred = le.inverse_transform(y_pred)
    y_true = le.inverse_transform(y_true)
    print_confusion_matrix(y_true, y_pred)
    print("Final Acc (valid): %.4f" % (acc))

#Function that merges the labels into 3 classes (neg,pos,neutral)
def label_merge(labels_raw):
    labels = []
    pos = []
    neg = []
    neutral = []
    for label in labels_raw:
        if label == "anger":
            labels.append("neg")
        elif label == "boredom":
            labels.append("neg")
        elif label == "hate":
            labels.append("neg")
        elif label == "sadness":
            labels.append("neg")
        elif label == "worry":
            labels.append("neg")
        elif label == "empty":
            labels.append("neutral")
        elif label == "enthusiasm":
            labels.append("pos")
        elif label == "fun":
            labels.append("pos")
        elif label == "happiness":
            labels.append("pos")
        elif label == "love":
            labels.append("pos")
        elif label == "relief":
            labels.append("pos")
        elif label == "surprise":
            labels.append("neutral")
        else:
            labels.append("neutral")
    for label in labels:
        if label == "neutral":
            neutral.append(label)
        elif label == "pos":
            pos.append(label)
        elif label == "neg":
            neg.append(label)
    print("label dist:")
    print("pos= ", len(pos))
    print("neg= ", len(neg))
    print("neutral= ", len(neutral))
    return labels


def encoding_labels(labels, le):
    labels_encoded = le.fit_transform(labels)
    return labels_encoded


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()

    pretrained = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    """ Loads in the twitter data"""
    df_train = pd.read_csv('data/train.csv')
    df_train.head()
    inputs_train = df_train["content"]
    labels_train = df_train["sentiment"]
    # For merged labels:
    # labels_train = label_merge(labels_train)
    le = LabelEncoder()
    labels_train = encoding_labels(labels_train, le)

    df_test = pd.read_csv('data/test.csv')
    df_test.head()
    inputs_test = df_test["content"]
    labels_test = df_test["sentiment"]
    # For merged labels:
    # labels_test = label_merge(labels_test)
    labels_test = encoding_labels(labels_test, le)

    train_dataset = Dataset(inputs_train, labels_train, tokenizer)

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=args.batch_size)
    valid_dataset = Dataset(inputs_test, labels_test, tokenizer)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size)

    # Load a pretrained BERTbase - uncased model
    model = BertForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=3,
        # num_hidden_layers=args.num_hidden_layers,
        # num_attention_heads=args.num_attn_heads,
        output_attentions=True)
    model = model.to(device)

    #training model
    train(model, train_loader, valid_loader, args.epochs, le)