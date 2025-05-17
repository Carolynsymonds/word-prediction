
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import torch
import pandas as pd
from transformers import BertTokenizer

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

class TextDatasetBert(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.LongTensor(sample[:-1])  # input: first N tokens
        target_seq = torch.LongTensor(sample[1:])  # target: next N tokens
        return input_seq, target_seq

def load_and_combine_snippets(filenames, column='snippet'):
    formatted_headlines = []

    for file in filenames:
        try:
            df = pd.read_csv(f"data/{file}", on_bad_lines='skip', usecols=[column], engine='python')
            print(f"Loaded {file} with {df.shape[0]} rows.")
            formatted_headlines.extend(df[column].dropna().tolist())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return formatted_headlines

def load_dataloader(sequence_length, batch_size, option=1):
    files = [
        'ArticlesMarch2018.csv',
        'ArticlesApril2017.csv',
        'ArticlesFeb2017.csv',
        'ArticlesFeb2018.csv',
        'ArticlesJan2017.csv',
        'ArticlesJan2018.csv',
        'ArticlesMarch2017.csv',
        'ArticlesApril2018.csv',
    ]
    formatted_headlines = load_and_combine_snippets(files)
    files = [
        # 'CommentsApril2017.csv',
        # 'CommentsApril2018_clean.csv',
        # 'CommentsFeb2017.csv',
        # 'CommentsFeb2018.csv',
        # 'CommentsJan2018.csv',
        # 'CommentsJan2017.csv',
        # 'CommentsMarch2017.csv',
        # 'CommentsMarch2018.csv',
        # 'CommentsMay2017.csv',
    ]

    formatted_articles_comments = load_and_combine_snippets(files, 'commentBody')
    # Not possible due to computational resources
    formatted_articles_comments = formatted_articles_comments[:len(formatted_articles_comments) // 5]

    combined_text = formatted_headlines + formatted_articles_comments

    print(f"Total combined headlines: {len(combined_text)}")

    paragraph = ' '.join(combined_text)
    dataset = ''
    # EVAL 1: Tokenizer by scratch
    if option == 1:
        words = paragraph.split()
        word_counts = Counter(words)
        vocab = list(word_counts.keys())
        vocab_size = len(vocab)
        word_to_int = {word: i for i, word in enumerate(vocab)}
        int_to_word = {i: word for word, i in word_to_int.items()}
        samples = [words[i:i + sequence_length + 1] for i in range(len(words) - sequence_length)]
        # print(vocab)
        print(word_to_int)
        print(int_to_word)
        dataset = TextDataset(samples, word_to_int)

    # EVAL 2: BERT tokenizer
    if option == 2:

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        tokens = tokenizer.tokenize(paragraph)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create vocabulary info from tokenizer
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        # Optionally: Sort and print the first 20 tokens
        for word, idx in list(tokenizer.vocab.items())[:100]:
            print(f"{word}: {idx}")

        # Map ids to tokens and vice versa
        word_to_int = vocab
        int_to_word = {v: k for k, v in vocab.items()}
        samples = [token_ids[i:i + sequence_length + 1] for i in range(len(token_ids) - sequence_length)]

        # Prints
        print("Sample tokens:", tokens[:10])
        print("Sample token IDs:", token_ids[:10])
        print("Vocab size:", vocab_size)

        dataset = TextDatasetBert(samples)
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