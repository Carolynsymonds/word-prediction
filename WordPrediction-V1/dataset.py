import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, random_split


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

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_dataloaders():
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

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the entire list
    encoded_inputs = tokenizer(
        formatted_headlines,
        padding=True,  # Pad to the longest sequence in the batch
        truncation=True,  # Truncate longer sequences
        return_tensors='pt'  # Return as PyTorch tensors
    )
    word_index = tokenizer.get_vocab()
    vocab_size = len(word_index) + 1
    max_seq_len = encoded_inputs['input_ids'].shape[1]
    print(f"Max sequence length: {max_seq_len}")

    input_ids = encoded_inputs['input_ids']
    X = input_ids[:, :-1]
    y = input_ids[:, 1:]

    idx = 0  # any index in your batch

    input_ids = X[idx].tolist()
    target_ids = y[idx].tolist()

    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
    decoded_target = tokenizer.decode(target_ids, skip_special_tokens=True)

    print("Input token IDs:  ", input_ids)
    print("Target token IDs: ", target_ids)
    print("\nDecoded Input:    ", decoded_input)
    print("Decoded Target:   ", decoded_target)

    for i in range(10):
        input_token = tokenizer.decode([X[0][i]])
        target_token = tokenizer.decode([y[0][i]])
        print(f"{input_token:>15} → {target_token}")

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader, vocab_size, tokenizer


