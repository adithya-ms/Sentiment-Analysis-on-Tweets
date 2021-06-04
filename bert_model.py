import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from baseline_models import splitData
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

parser = argparse.ArgumentParser(description="Sentiment analysis")
parser.add_argument("--reload_model", type=str, help="Path of model to reload")
parser.add_argument("--save_model", type=str, help="Path for saving the model")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
# parser.add_argument("--num_hidden_layers", type=int, default=1)
# parser.add_argument("--num_attn_heads", type=int, default=1)


class Dataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        super().__init__()

        #encoding labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        # Making lists for: labels, input_ids, attention_masks
        labels = []
        input_ids = []
        attention_masks = []
        for line, label in zip(data,labels_encoded):
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
        return self.input_ids[index].long(),self.attention_masks[index].long(), self.labels[index].long()

    def __len__(self):
        return len(self.input_ids)



def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        correct = 0.0 
        total = 0.0
        for batch in loader:
            input_ids, attention_masks, labels = batch
            out = model(input_ids=input_ids, token_type_ids=None, attention_mask= attention_masks,labels=labels,return_dict=True)
            preds = out.logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.numel()


    print("predicited ",correct, " out of ", total, " Correctly")
    model.train()
    return correct/total

def train(model, train_loader, valid_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0.0
        for i, batch in enumerate(train_loader):

            if i % 5 == 0 and i > 0:
                tps = total_tokens/ (time.time()-start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, toks/sec: %d" % (epoch, i, total_loss/5, tps))
                start_time = time.time()
                total_loss = 0.0
                total_tokens = 0.0

            input_ids, attention_masks, labels = batch
            optimizer.zero_grad()
            out = model(input_ids=input_ids,token_type_ids=None, attention_mask=attention_masks, labels=labels, return_dict=True)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
            total_tokens += input_ids.numel()

        acc = evaluate(model, valid_loader)
        print("[Epoch %d] Acc (valid): %.4f" % (epoch, acc))
        start_time = time.time() # so tps stays consistent

    print("############## END OF TRAINING ##############")
    acc = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (acc))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()
    args.save_model = "model.pt"
    
    pretrained = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = True

    #########
    """ Loads in the twitter data"""
    df = pd.read_csv('data/text_emotion.csv')
    df.head()
    data = df["content"]
    labels = df["sentiment"]

    """ Preprocessing (not yet implemented) """

    """ Baseline implementation"""
    inputs_train, inputs_test, labels_train, labels_test = splitData(data, labels)
    #########

    train_dataset = Dataset(inputs_train, labels_train, tokenizer)

    train_loader = DataLoader(train_dataset,
        shuffle=True,
        batch_size=args.batch_size)
    valid_dataset = Dataset(inputs_test, labels_test, tokenizer)
    valid_loader = DataLoader(valid_dataset,
        batch_size=args.batch_size)

    # # test_dataset = POSDataset("data/test.en", "data/test.en.label", tokenizer)
    # test_dataset = POSDataset("data/test.en", None, tokenizer)
    # test_loader = DataLoader(test_dataset,
    #     batch_size=args.batch_size)
    #

    # Load a pretrained model
    model = BertForSequenceClassification.from_pretrained(
        pretrained, 
        num_labels=13,
        # num_hidden_layers=args.num_hidden_layers,
        # num_attention_heads=args.num_attn_heads,
        output_attentions=True)

    # Load model weights from a file
    # args.reload_model = "model.pt"
    # if args.reload_model:
    #     model.load_state_dict(torch.load(args.reload_model))

    train(model, train_loader, valid_loader, args.epochs)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
