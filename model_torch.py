
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) #NEW
        self.bilstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(lstm_dim * 2, 100)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(in_features=100, out_features=vocab_size)

    def forward(self, inputs):
        x = self.embedding(inputs)  # inputs shape: (batch_size, sequence_length)
        x, _ = self.bilstm(x)  # x: (batch_size, seq_len, 2 * lstm_dim)
        x, (h_n, c_n) = self.lstm(x)  # x: (batch_size, seq_len, 100)
        last_hidden = x[:, -1, :]    # shape: (batch_size, 100)
        x = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output
