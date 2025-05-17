import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import AttentionLSTMLanguageModel
from utils import load_config, save_checkpoint, plot_metrics
from metrics import MetricsLogger
from tqdm import tqdm
import numpy as np
import nltk
from transformers import BertTokenizer
import matplotlib.pyplot as plt

#function to train model
def train_model(config_path):
    config = load_config(config_path)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, vocab, _ = get_dataloaders(
        batch_size=config['batch_size'],
        curriculum=config.get('curriculum', False),
        seq_len=config.get('seq_len', 10)
    )

    model = AttentionLSTMLanguageModel(
        tokenizer=vocab.tokenizer,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    metrics_logger = MetricsLogger()

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss, total_correct, total_tokens = 0, 0, 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, attn_weights = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.size(0)

        train_acc = total_correct / total_tokens
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_correct, val_total, val_topk = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, attn_weights = model(inputs)
                loss = loss_fn(logits, targets)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                val_topk += (logits.topk(5, dim=1).indices == targets.unsqueeze(1)).any(dim=1).sum().item()

        val_acc = val_correct / val_total
        val_top5 = val_topk / val_total
        avg_val_loss = val_loss / len(val_loader)
        perplexity = np.exp(avg_val_loss)

        metrics_logger.log_epoch(avg_train_loss, avg_val_loss, train_acc, val_acc, val_top5, perplexity)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
              f"Train Acc = {train_acc:.2%}, Val Acc = {val_acc:.2%}, Top-5 Acc = {val_top5:.2%}, PPL = {perplexity:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, metrics_logger.get_metrics_history(),
                            os.path.join(config['checkpoint_dir'], "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        plot_metrics(metrics_logger.get_metrics_history(), config['output_dir'])

    # Save metric plots at the end of training
    history = metrics_logger.get_metrics_history()
    epochs = list(range(1, len(history['train_loss']) + 1))

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig("accuracy_curve.png")

    plt.figure()
    plt.plot(epochs, history['val_top5'], label='Top-5 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Accuracy')
    plt.title('Top-5 Accuracy over Epochs')
    plt.legend()
    plt.savefig("top5_accuracy_curve.png")

    plt.figure()
    plt.plot(epochs, history['perplexity'], label='Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity over Epochs')
    plt.legend()
    plt.savefig("perplexity_curve.png")

    # Inference (next word prediction)
    prompt = "ban hurts universities and"
    model.eval()
    with torch.no_grad():
        tokenizer = vocab.tokenizer
        tokens = tokenizer.tokenize(prompt.lower())
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([ids]).to(device)

        logits, attn_weights = model(input_tensor)
        predicted_id = logits[0].argmax(dim=0).item()
        predicted_word = tokenizer.convert_ids_to_tokens([predicted_id])[0]

        print("Prompt:", prompt)
        print("Predicted next word:", predicted_word)
        print("Attention weights:", attn_weights.squeeze(0).cpu().numpy())


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "flatconfig.yaml")
    train_model(config_path)