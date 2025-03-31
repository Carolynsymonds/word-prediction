import numpy as np
import os, sys, re
import torch
import torch.nn as nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()

class RNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function

        # define embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # define lstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # define model layers
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function
        batch_size = x.size(0)
        x = x.long()

        # embedding and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm layers
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout, fc layer and final sigmoid layer
        out = self.fc(lstm_out)

        # reshaping out layer to batch_size * seq_length * output_size
        out = out.view(batch_size, -1, self.output_size)

        # return last batch
        out = out[:, -1]

        # return one batch of output word scores and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weights.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        # initialize hidden state with zero weights, and move to GPU if available

        return hidden

