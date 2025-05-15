import yaml
import json
import matplotlib.pyplot as plt
import os

def plot_metrics(metrics_history, output_dir):
    # Create directory for metrics plots
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, 'loss_curves.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['train_acc'], label='Training Accuracy')
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, 'accuracy_curves.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['val_topk_acc'], label='Validation Top K Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Top K Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, 'topk_accuracy_curves.png'))
    plt.close()

    with open(os.path.join(metrics_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=4)

def load_config(config_path):
    # Load YAML configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
