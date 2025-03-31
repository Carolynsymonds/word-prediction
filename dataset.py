from torch.utils.data import Dataset
import torch
import pandas as pd
import re
import unicodedata


class MediumDataset(Dataset):
    MAX_LENGTH = 10
    SOS_token = 0
    EOS_token = 1
    UNK_token = 2

    def __init__(self, vocab=None):
        self.df = pd.read_csv("medium_data.csv")
        self.df['title'] = self.df['title'].astype(str).fillna('')
        self.vocab = vocab or self.build_vocab()
        self.word2idx = self.vocab
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        self.samples = self.build_sequences()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def encode(self, tokens):
        return [self.vocab.get(token, self.UNK_token) for token in tokens]

    def build_sequences(self):
        samples = []
        for title in self.df['title']:
            title = self.normalize_string(title)
            words = title.split()
            words = words[:self.MAX_LENGTH - 1]  # leave room for EOS
            words.append('<eos>')
            token_ids = self.encode(words)
            for i in range(1, len(token_ids)):
                input_seq = token_ids[:i]
                target = token_ids[i]
                samples.append((input_seq, target))
        return samples


    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def build_vocab(self):
        vocab = {'<sos>': self.SOS_token, '<eos>': self.EOS_token, '<unk>': self.UNK_token}
        idx = 3
        for title in self.df['title']:
            title = self.normalize_string(title)
            words = title.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab