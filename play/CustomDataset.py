from torch.utils.data import Dataset
import torch
import pandas as pd
import re
import unicodedata


class CustomDataset(Dataset):

  MAX_LENGTH = 10
  SOS_token = 0
  EOS_token = 1
  UNK_token = 2

  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def normalize_string(self, s):
      s = self.unicode_to_ascii(s.lower().strip())
      s = re.sub(r"([.!?])", r" \1", s)
      s = re.sub(r"[^a-zA-Z!?]+", r" ", s)

  def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

  def build_vocab(self):
      vocab = {'<sos>': self.SOS_token, '<eos>': self.EOS_token, '<unk>': self.UNK_token}
      idx = 3
      for title in self.X:
          title = self.normalize_string(title)
          words = title.split()
          for word in words:
              if word not in vocab:
                  vocab[word] = idx
                  idx += 1
      return vocab