
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import torch
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, samples, word_to_int):
        self.samples = samples
        self.word_to_int = word_to_int
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.LongTensor([self.word_to_int[word] for word in sample[:-1]])
        target_seq = torch.LongTensor([self.word_to_int[word] for word in sample[1:]])
        return input_seq, target_seq

def load_and_combine_snippets(filenames, column='snippet'):
    formatted_headlines = []

    for file in filenames:
        try:
            df = pd.read_csv(f"data/{file}")
            print(f"Loaded {file} with {df.shape[0]} rows.")
            formatted_headlines.extend(df[column].dropna().tolist())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    print(f"Total combined headlines: {len(formatted_headlines)}")
    return formatted_headlines

def load_dataloader(sequence_length, batch_size):
    files = [
        'ArticlesApril2017.csv'
    ]

    formatted_headlines = load_and_combine_snippets(files)
    paragraph = ' '.join(formatted_headlines)
    words = paragraph.split()

    word_counts = Counter(words)
    vocab = list(word_counts.keys())
    vocab_size = len(vocab)
    word_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_word = {i: word for word, i in word_to_int.items()}
    samples = [words[i:i + sequence_length + 1] for i in range(len(words) - sequence_length)]
    print(vocab)
    print(word_to_int)
    print(int_to_word)

    dataset = TextDataset(samples, word_to_int)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataloader, val_dataloader, vocab_size, int_to_word