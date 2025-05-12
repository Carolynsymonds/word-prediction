import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_output))

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim=200, num_heads=4, ff_dim=128, dropout=0.2):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.embedding(x)                          # (B, T, D)
        x = self.transformer(x)                        # (B, T, D)
        x = self.norm(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = x.reshape(-1, 512)  # (B * T, 512)
        return F.log_softmax(self.output_layer(x), dim=-1)