import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import Counter
import unicodedata
import re
import torch
import os
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader, random_split, TensorDataset
from model_torch import LSTMModel
import string


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
def plot(train_losses, val_losses, val_accuracy):
    # Plot Train loss
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.savefig('train_loss.png')
    plt.show()

    # Plot Val loss
    plt.plot(range(1, len(val_losses) + 1), val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.grid(True)
    plt.savefig('val_loss.png')
    plt.show()

    # Plot Val accuracy
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    plt.savefig('val_accuracy.png')
    plt.show()

def save_model(filename, model):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model.state_dict(), save_filename)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def tokenize(data):
    # Clean and tokenize the text
    titles = data['title'].astype(str).apply(lambda x: x.lower())  # lowercase
    all_tokens = []

    for title in titles:
        title = normalize_string(title)
        tokens = word_tokenize(title)
        all_tokens.extend(tokens)

    # Count token frequency
    token_counts = Counter(all_tokens)

    # Create word index with <OOV> token
    word_index = {'<OOV>': 1}
    for i, (word, count) in enumerate(token_counts.items(), start=2):  # start=2 because 1 is <OOV>
        word_index[word] = i

    # Final vocab size
    vocab_size = len(word_index) + 1  # +1 for padding index (usually 0)

    return vocab_size, word_index
def tokenize2(data):
    # Clean and tokenize the text
    data = data.lower()  # lowercase
    all_tokens = []

    data = normalize_string(data)
    tokens = word_tokenize(data)
    all_tokens.extend(tokens)

    # Count token frequency
    token_counts = Counter(all_tokens)

    # Create word index with <OOV> token
    word_index = {'<OOV>': 1}
    for i, (word, count) in enumerate(token_counts.items(), start=2):  # start=2 because 1 is <OOV>
        word_index[word] = i

    # Final vocab size
    vocab_size = len(word_index) + 1  # +1 for padding index (usually 0)

    return vocab_size, word_index

def text_to_sequence(text, word_index):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [word_index.get(word, word_index['<OOV>']) for word in tokens]

def keras_like_pad_sequences(sequences, maxlen=None, padding_value=0):
    if not maxlen:
        maxlen = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        pad_len = maxlen - len(seq)
        padded_seq = [padding_value] * pad_len + seq  # pre-padding
        padded.append(padded_seq)

    return np.array(padded)

def train(epochs, train_loader, val_loader, model, optimizer, criterion, word_index):
    train_epoch_losses, val_epoch_losses, val_epoch_accuracy = [], [], []
    valid_loss_min = np.Inf

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

        for batch_idx, (inputs, target) in enumerate(progress_bar):
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(inputs)

            loss = criterion(output, target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            #Loss and Accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            #Decode words for comparison
            index_to_word = {index: word for word, index in word_index.items()}

            if batch_idx == 0:
                print("\n--- First Batch Debug ---")
                print("Input (token IDs):", inputs[0])  # first sequence in batch
                print("Target:", target[0])
                print("Predicted:", predicted[0])

                decoded_input = [index_to_word.get(idx.item(), ' ') for idx in inputs[0]]
                decoded_target = index_to_word.get(target[0].item(), ' ')
                decoded_prediction = index_to_word.get(predicted[0].item(), ' ')

                print("Decoded input:", " ".join(decoded_input))
                print("Decoded target word:", decoded_target)
                print("Decoded prediction:", decoded_prediction)
                print("-------------------------\n")


        avg_train_loss = train_loss / total
        train_accuracy = correct / total * 100
        train_epoch_losses.append(avg_train_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for val_input, val_target in val_loader:

                val_input, val_target = val_input.to(device), val_target.to(device)

                val_output = model(val_input)

                #Loss and Accuracy
                loss = criterion(val_output, val_target)
                val_loss += loss.item() * val_input.size(0)
                _, predicted = torch.max(val_output, dim=1)
                correct += (predicted == val_target).sum().item()
                total += val_target.size(0)

        avg_val_loss = val_loss / total
        val_accuracy = correct / total * 100
        val_epoch_losses.append(avg_val_loss)
        val_epoch_accuracy.append(val_accuracy)

        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if avg_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), 'state_dict.pt')
            print(f"Validation loss decreased ({valid_loss_min:.6f} --> {avg_val_loss:.6f}). Saving model ...")
            valid_loss_min = avg_val_loss

    plot(train_epoch_losses, val_epoch_losses, val_epoch_accuracy)
    return model
def load_and_combine_snippets(filenames, column='snippet'):
    print("Loading and cleaning data...")

    formatted_headlines = []

    for file in filenames:
        try:
            df = pd.read_csv(f"Transformers/data/{file}")
            print(f"Loaded {file} with {df.shape[0]} rows.")
            formatted_headlines.extend(df[column].dropna().tolist())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    print(f"Total combined headlines: {len(formatted_headlines)}")
    return formatted_headlines
def remove_puntuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
def main():
    print("Loading and cleaning data...")
    data = pd.read_csv('./medium_data.csv')
    files = [
        'ArticlesMarch2018.csv',
        'ArticlesApril2017.csv'
    ]

    formatted_headlines = load_and_combine_snippets(files)
    formated_text = '\n'.join(formatted_headlines)
    formated_text = remove_puntuations(formated_text)
    formated_text = formated_text.lower()

    print(f"Original records: {data.shape[0]}")

    data.dropna(subset=['title'], inplace=True)
    data.drop_duplicates(subset=['title'], inplace=True)
    data['title'] = data['title'].apply(lambda x: x.replace(u'\xa0', ' ').replace('\u200a', ' '))

    print("Tokenizing text...")
    vocab_size, word_index = tokenize(data)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Cleaned records: {data.shape[0]}")

    print("Tokenizing text...")
    vocab_size, word_index = tokenize2(formated_text)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Cleaned records: {data.shape[0]}")

    sequences = []
    for line in tqdm(formated_text.split('\n'), desc="Generating sequences"):
        token_list = text_to_sequence(line, word_index)  # manual text to sequence

        for i in range(1, len(token_list)):
            seq = token_list[:i + 1]
            sequences.append(seq)


    print(f"Total sequences: {len(sequences)}")
    max_seq_len = max([len(seq) for seq in sequences])
    padded = keras_like_pad_sequences(sequences)

    print("Creating features and labels...")
    X, y = padded[:, :-1], padded[:, -1]
    y = torch.tensor(y, dtype=torch.long)  # Convert to PyTorch tensor and ensure it's int64

    print("Building the arq...")
    model = LSTMModel(vocab_size, 128, 150).to(device)
    print(model)
    dummy_input = torch.randint(0, vocab_size, (32, max_seq_len), dtype=torch.long).to(device)
    summary(model, input_data=dummy_input)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    trained_rnn = train(epochs=40,train_loader=train_loader, val_loader=val_loader, model=model, optimizer=optimizer, criterion=criterion, word_index=word_index)

    save_model('trained_rnn', trained_rnn)
    print('Model Trained and Saved')

#TBD
    print("Generating text with top-3 suggestions...")
    seed_text = "implementation of"
    num_words = 3

    for _ in range(num_words):
        token_list = text_to_sequence(seed_text, word_index)  # manual text to sequence
        padded = keras_like_pad_sequences([token_list])

        prediction = model.predict(padded, verbose=0)[0]

        top_indices = prediction.argsort()[-3:][::-1]
        top_words = [word for word, index in word_index.items() if index in top_indices]

        seed_text += " " + top_words[0]  # Pick the best one
        print(f"Top suggestions: {top_words}")

    print("Final generated text:")
    print(seed_text)

if __name__ == "__main__":
    main()
