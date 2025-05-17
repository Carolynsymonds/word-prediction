import torch
import torch.nn as nn

class AttentionLSTMLanguageModel(nn.Module):
    def __init__(self, tokenizer, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_idx = tokenizer.pad_token_id

        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)        
        lstm_out, _ = self.lstm(embedded)           

        # Use last hidden state as query
        last_output = lstm_out[:, -1:, :]           
        energy = torch.tanh(self.attn(lstm_out))    
        attn_scores = torch.bmm(energy, last_output.transpose(1, 2))  
        attn_weights = torch.softmax(attn_scores, dim=1)          
        context = (attn_weights * lstm_out).sum(dim=1)             

        logits = self.fc(context)         
        return logits, attn_weights  # return attention for debugging