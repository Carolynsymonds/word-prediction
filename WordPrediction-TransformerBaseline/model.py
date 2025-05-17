import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads,sequence_length):
        super(TextGen, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=sequence_length, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        #OPTION 1
        # self.decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=embed_dim,
        #     nhead=num_heads,
        #     batch_first=True
        # )
        # OPTION 2
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    # Positional encoding is required. Else the model does not learn.
    def forward(self, x):
        emb = self.emb(x)
        x = self.pos_encoder(emb)

        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        for block in self.transformer_blocks:
            x = block(x, src_mask=input_mask)
        # x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)

        return out

# class TextGen(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_layers, num_heads,sequence_length):
#         super(TextGen, self).__init__()
#         self.pos_encoder = PositionalEncoding(max_len=sequence_length, d_model=embed_dim)
#         self.emb = nn.Embedding(vocab_size, embed_dim)
#         self.decoder_layer = nn.TransformerDecoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(
#             decoder_layer=self.decoder_layer,
#             num_layers=num_layers,
#         )
#         self.linear = nn.Linear(embed_dim, vocab_size)
#         self.dropout = nn.Dropout(0.2)
#
#     # Positional encoding is required. Else the model does not learn.
#     def forward(self, x):
#         emb = self.emb(x)
#
#         # Generate input sequence mask with shape (SEQUENCE_LENGTH, SEQUENCE_LENGTH)
#         input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
#
#         x = self.pos_encoder(emb)
#         x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
#         x = self.dropout(x)
#         out = self.linear(x)
#         return out
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, T, T]

        if mask is not None:
            # Ensure mask is expanded to match [B, T, T]
            mask = mask.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]
            # Clamp to a large negative number (not -inf)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        attn_output = torch.matmul(attn_weights, V)
        return self.out_proj(attn_output)

class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = SingleHeadSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x
class TextGenSingleAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, sequence_length):
        super(TextGenSingleAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=sequence_length, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)

        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(embed_dim) for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    # Positional encoding is required. Else the model does not learn.
    def forward(self, x):
        emb = self.emb(x)
        x = self.pos_encoder(emb)

        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        for block in self.transformer_blocks:
            x = block(x, mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)

        return out
