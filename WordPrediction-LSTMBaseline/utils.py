import os
import json
import yaml
import torch
import matplotlib.pyplot as plt

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def plot_metrics(metrics, output_dir):
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    for key, values in metrics.items():
        plt.figure()
        plt.plot(values)
        plt.title(key)
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.grid(True)
        plt.savefig(f"{output_dir}/metrics/{key}.png")
        plt.close()
