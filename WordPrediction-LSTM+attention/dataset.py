import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

#Vocab class
class Vocab:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.unk_idx = tokenizer.unk_token_id

    def encode(self, text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def decode(self, indices):
        return self.tokenizer.convert_ids_to_tokens(indices)

    def __len__(self):
        return self.tokenizer.vocab_size

#Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, vocab, seq_len=10):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = []

        for line in texts:
            token_ids = vocab.encode(line)
            if len(token_ids) <= seq_len:
                continue
            for i in range(len(token_ids) - seq_len):
                X = token_ids[i:i + seq_len]
                y = token_ids[i + seq_len]
                if y != vocab.unk_idx:
                    self.data.append((X, y))

        print(f"Total sequences: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)


def collate_batch(batch):
    X, y = zip(*batch)
    X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return X_padded, y


def get_dataloaders(batch_size=32, curriculum=False, seq_len=10):
    path = os.path.join(os.path.dirname(__file__), 'data', 'CommentsMay2017.csv')
    if not os.path.exists(path):
        raise FileNotFoundError("CommentsMay2017.csv not found")

    df = pd.read_csv(path)
    df = df.dropna(subset=['commentBody']).head(50000)  
    texts = df['commentBody'].astype(str).tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = Vocab(tokenizer)
    dataset = NewsDataset(texts, vocab, seq_len=seq_len)

    train_size = int(0.8 * len(dataset))
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_batch)

    return train_loader, val_loader, vocab, None
