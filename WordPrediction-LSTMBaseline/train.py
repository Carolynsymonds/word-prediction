# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import LSTMLanguageModel
from utils import setup_device, load_config, save_checkpoint, plot_metrics
from metrics import MetricsLogger
from logger import Logger
from tqdm import tqdm
import os

#train model function
def train_model(config_path):
    config = load_config(config_path)
    device = setup_device()

    train_loader, val_loader, vocab_size, tokenizer = get_dataloaders()
    model = LSTMLanguageModel(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))

    loss_fn = nn.CrossEntropyLoss()
    metrics_logger = MetricsLogger()

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss, total_tokens, total_correct = 0, 0, 0
        for inputs, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, target.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct = (pred == target.view(-1)).sum().item()
            total_correct += correct
            total_tokens += target.numel()

        train_acc = total_correct / total_tokens
        #eval of the model
        model.eval()
        with torch.no_grad():
            total_val_loss, val_correct, val_tokens, topk_correct, perplexity_sum = 0, 0, 0, 0, 0
            for inputs, target in val_loader:
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, target.view(-1))
                total_val_loss += loss.item()

                pred = outputs.argmax(dim=1)
                val_correct += (pred == target.view(-1)).sum().item()
                val_tokens += target.numel()

                top_k_preds = outputs.topk(5, dim=1).indices
                topk_correct += top_k_preds.eq(target.view(-1).unsqueeze(1)).any(dim=1).sum().item()
                perplexity_sum += torch.exp(loss).item()

        val_acc = val_correct / val_tokens
        val_topk_acc = topk_correct / val_tokens
        avg_val_loss = total_val_loss / len(val_loader)
        avg_train_loss = total_train_loss / len(train_loader)
        avg_perplexity = perplexity_sum / len(val_loader)

        metrics_logger.log_epoch(avg_train_loss, avg_val_loss, train_acc, val_acc, val_topk_acc, avg_perplexity)
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, "
            f"Train Acc = {train_acc:.2%}, "
            f"Val Acc = {val_acc:.2%}, "
            f"Top-5 Acc = {val_topk_acc:.2%}, "
            f"PPL = {avg_perplexity:.2f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config['output_dir'], exist_ok=True)
            save_checkpoint(model, optimizer, epoch, metrics_logger.get_metrics_history(), f"{config['output_dir']}/best_lstm.pth")

        plot_metrics(metrics_logger.get_metrics_history(), config['output_dir'])

        # Final evaluation after training
        model.eval()
        prompt = "The president said that"
        max_words = 3

        tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(device)

        generated = input_ids[:, :-1]  
        print(f"\nPrompt: {prompt}")

        for _ in range(max_words):
            with torch.no_grad():
                output = model(generated)
            next_token_logits = output[-1] 
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)  
            generated = torch.cat([generated, next_token_id], dim=1)

            next_word = tokenizer.decode([next_token_id.item()])
            print(f"Predicted word: {next_word}")


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "flatconfig.yaml")

    config = load_config(config_path)
    print("Loaded config:\n", config)
    train_model(config_path)  