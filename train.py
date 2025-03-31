from dataset import MediumDataset
from torch.utils.data import DataLoader, random_split
from models import RNN
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import save_checkpoint, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = [torch.tensor(seq, dtype=torch.long) for seq in inputs]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # 0 = <pad>
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs_padded, targets

def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param hidden:
    :param optimizer:
    :param rnn:
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """

    # TODO: Implement Function

    # move data to GPU, if available
    rnn.to(device)

    # creating variables for hidden state to prevent back-propagation
    # of historical states
    h = tuple([each.data for each in hidden])

    rnn.zero_grad()
    # move inputs, targets to GPU
    inputs, targets = inp.to(device), target.to(device)

    output, h = rnn(inputs, h)

    loss = criterion(output, targets)

    # perform backpropagation and optimization
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h

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
    num_epochs = 5
    # Learning Rate
    learning_rate = 0.001
    # Embedding Dimension
    embedding_dim = 200
    # Hidden Dimension
    hidden_dim = 250
    # Number of RNN Layers
    n_layers = 2
    batch_size=20
    # Show stats for every n number of batches
    show_every_n_batches = 500

    dataset = MediumDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    vocab_size = len(dataset.vocab)
    output_size = vocab_size

    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    rnn.to(device)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    trained_rnn, train_losses, val_losses, val_accuracy = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, train_loader, val_loader, show_every_n_batches)

    plot(train_losses, val_losses, val_accuracy)
    save_model('/save/trained_rnn-test', trained_rnn)
    print(trained_rnn)
    print('Model Trained and Saved')

def plot(train_losses, val_losses, val_accuracy):
    # Plot Train loss
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.show()
    # Plot Val loss
    plt.plot(range(1, len(val_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.grid(True)
    plt.show()
    # Plot Val accuracy
    plt.plot(range(1, len(val_accuracy) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main('config.yaml')

