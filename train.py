import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle as pickle

from models.model import *
from util import *
import matplotlib
import matplotlib.pyplot as plt


# Load vocabulary and word_to_ix
words = pickle.load(open(f'./pretrained_word_embeddings/GloVe/6B.50_words.pkl', 'rb'))
word_to_ix = pickle.load(open(f'./pretrained_word_embeddings/GloVe/6B.50_word_idx.pkl', 'rb'))
weights_matrix = pickle.load(open(f'./pretrained_word_embeddings/GloVe/6B.50_weights_matrix.pkl', 'rb'))

# Load training data
training_sentences, training_tags = select_dataset("./data/train.sents.txt", "./data/train.tags.txt")
training_data = load_train_dataset(training_sentences, training_tags)
print("We have %i training data" % (len(training_data)))

# Set parameters
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50
HIDDEN_DIM = 10
POS_EMBEDDING_DIM = 5

# Stanovsky
tag_to_ix = {"A0-B": 0, "A0-I": 1, "A1-B": 2, "A1-I": 3, "A2-B": 4, "A2-I": 5, "A3-B": 6, "A3-I": 7, "A4-B": 8, "A4-I": 9, "A5-B": 10, "A5-I": 11, "P-B": 12, "P-I": 13, "O": 14, START_TAG: 15, STOP_TAG: 16}
ix_to_tag = {0: "A0-B", 1: "A0-I", 2: "A1-B", 3: "A1-I", 4: "A2-B", 5: "A2-I", 6: "A3-B", 7: "A3-I", 8: "A4-B", 9: "A4-I", 10: "A5-B", 11: "A5-I", 12: "P-B", 13: "P-I", 14: "O", 15: START_TAG, 16: STOP_TAG}

# Training the model
model = BiLSTM_CRF(len(words), tag_to_ix, weights_matrix, EMBEDDING_DIM, HIDDEN_DIM, POS_EMBEDDING_DIM, START_TAG, STOP_TAG)
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype = torch.long)
    print(model(precheck_sent))

mean_cost = []

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(50):
    epoch_costs = []
    
    print("Starting epoch %i..." % (epoch))
    for sentence, tags in training_data:
        # 1. Clear the accumulate gradients
        model.zero_grad()
        
        # 2. Get inputs and transfer to tensor of index
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype = torch.long)
        
        # 3. Run the forward pass
        loss = model.neg_log_likelihood(sentence_in, targets)
        epoch_costs.append(loss.data.numpy())
        
        # 4. Compute loss, gradients, update parameters
        loss.backward()
        optimizer.step()
    mean_cost.append(np.mean(epoch_costs))
    print("Epoch %i, cost average: %s" % (epoch, np.mean(epoch_costs)))

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
    
print("Loss in 50 epochs: ", mean_cost)
epoch_num = [i for i in range(50)]
plt.plot(epoch_num, mean_cost)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()