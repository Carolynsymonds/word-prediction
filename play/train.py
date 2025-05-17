from dataset import MediumDataset
from torch.utils.data import DataLoader, random_split
from utils import collate_fn, forward_back_prop
from models import RNN, LSTMModel
import torch
import torch.nn as nn
import numpy as np

from utils import plot, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train2(epochs, train_loader, val_loader, model, optimizer, criterion):
    train_epoch_losses = []
    val_epoch_losses = []
    val_epoch_accuracy = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_epoch_losses.append(avg_train_loss)

        # -----------------------
        # 🔍 Validation loop
        # -----------------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)

                val_output = model(val_x)
                loss = criterion(val_output, val_y)
                val_loss += loss.item()

                _, predicted = torch.max(val_output, 1)
                correct += (predicted == val_y).sum().item()
                total += val_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_epoch_losses.append(avg_val_loss)

        val_accuracy = correct / total
        val_epoch_accuracy.append(val_accuracy)

        print(f"Epoch: {epoch + 1}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")

    plot(train_epoch_losses, val_epoch_losses, val_epoch_accuracy)
    return model

# Function to calculate accuracy
def calculate_accuracy(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to compute gradients
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Get model predictions
            outputs = model(batch_x)

            # Get the predicted word indices
            _, predicted = torch.max(outputs, dim=1)

            # Compare with actual labels
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total * 100
    return accuracy

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, train_loader, val_loader, show_every_n_batches=100):
    rolling_loss = []
    train_epoch_losses = []
    val_epoch_losses = []
    val_epoch_accuracy = []

    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)

        epoch_loss = 0
        # make sure you iterate over completely full batches, only
        n_batches = len(train_loader.dataset) // batch_size

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            if batch_i > n_batches:
                break
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
            # record loss
            rolling_loss.append(loss)
            epoch_loss += loss

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(rolling_loss)))
                rolling_loss = []

        avg_epoch_loss = epoch_loss / n_batches
        train_epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch_i} Training Loss: {avg_epoch_loss:.4f}")

        val_loss, accuracy = validate(rnn, batch_size, val_loader, criterion)
        val_epoch_losses.append(val_loss)
        val_epoch_accuracy.append(accuracy)
        print(f"Epoch {epoch_i} Validation Loss: {val_loss:.4f} Validation Accuracy: {accuracy:.4f}")

    # returns a trained rnn
    return rnn, train_epoch_losses, val_epoch_losses, val_epoch_accuracy

def validate(rnn, batch_size, val_loader, criterion):
    rnn.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Dynamically set batch size
            current_batch_size = inputs.size(0)
            hidden = rnn.init_hidden(current_batch_size)

            # Move hidden to correct device
            hidden = tuple([h.to(device) for h in hidden])

            # Forward pass | Prediction
            output, hidden = rnn(inputs, hidden)

            # Loss
            loss = criterion(output, targets)
            total_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    rnn.train()
    return avg_loss, accuracy

def main(config_path):
# Training parameters _ DEFINE IN FILE
    # Number of Epochs
    num_epochs = 50
    # Learning Rate
    learning_rate = 0.0005
    # Embedding Dimension
    embedding_dim = 150
    # Hidden Dimension
    hidden_dim = 200
    # Number of RNN Layers
    n_layers = 2
    batch_size=32
    dropout = 0.5
    # Show stats for every n number of batches
    show_every_n_batches = 500

    dataset = MediumDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(dataset.vocab)
    output_size = vocab_size

    # rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

    rnn = LSTMModel(vocab_size, embedding_dim, hidden_dim, n_layers, dropout=dropout)
    rnn.to(device)
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # trained_rnn, train_losses, val_losses, val_accuracy = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, train_loader, val_loader, show_every_n_batches)
    trained_rnn = train2(num_epochs, train_loader, val_loader, rnn, optimizer, criterion)

    # plot(train_losses, val_losses, val_accuracy)
    save_model('trained_rnn-test3', trained_rnn)
    print(trained_rnn)
    print('Model Trained and Saved')

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    accuracy = calculate_accuracy(trained_rnn, test_loader, device)
    print(f"Model Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    # main('config.yaml')

    dataset = MediumDataset()

    vocab_size = len(dataset.vocab)
    learning_rate = 0.0005
    # Embedding Dimension
    embedding_dim = 150
    # Hidden Dimension
    hidden_dim = 200
    # Number of RNN Layers
    n_layers = 2
    batch_size = 32
    dropout = 0.5
    output_size = vocab_size

    test_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    model.load_state_dict(torch.load('trained_rnn-test3.pt', map_location=device))
    model.to(device)

    accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Model Accuracy: {accuracy:.2f}%")
