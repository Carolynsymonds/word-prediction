import torch
import yaml
import os
import argparse
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, f"{model_name}_ckpt_{str(epoch)}.pth")


def load_checkpoint(model, file_name):
    ckpt = torch.load(file_name, map_location=device)
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights)
    print("Model's pretrained weights loaded!")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

def save_model(filename, model):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model.state_dict(), save_filename)

def plot(train_losses, val_losses, val_accuracy):
    # Plot Train loss
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.show()
    # Plot Val loss
    plt.plot(range(1, len(val_losses) + 1), val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.grid(True)
    plt.show()
    # Plot Val accuracy
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    plt.show()

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