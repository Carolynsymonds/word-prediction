from utils import setup_device, load_config
import os
from models import TransformerLanguageModel
import torch
import torch.optim as optim
from logger import Logger
from dataset import get_dataloaders
from tqdm import tqdm
import torch.nn as nn
from metrics import MetricsLogger
import ssl
from utils import save_checkpoint, plot_metrics
ssl._create_default_https_context = ssl._create_unverified_context

def train_model(config_path):
    # Setup
    config = load_config(config_path)
    device = setup_device()

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)

    # Initialize model, optimizer, and data loaders

    train_loader, val_loader, vocab_size, tokenizer = get_dataloaders()
    model = TransformerLanguageModel(vocab_size=vocab_size, max_seq_len=50)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    metrics_logger = MetricsLogger()


    # wandb_logger = Logger(
    #     "word-prediction-transformers{}_lr{}_weight_decay{}".format(
    #         config['batch_size'],
    #         config['learning_rate']
    #     ))
    # logger = wandb_logger.get_logger()
    # logger.watch(model, log='all', log_freq=100)
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss = 0
        total_tokens = 0
        total_correct = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

        for batch_idx, (inputs, target) in enumerate(progress_bar):
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Grab one sample
            input_ids = inputs[0]
            target_ids = target[0]
            # Compare and print
            if batch_idx == 0:
                print("\n--- Top-5 Prediction Comparison (Input → Target vs Top-5) ---")

                for i in range(len(input_ids)):
                    if target_ids[i].item() == tokenizer.pad_token_id:
                        continue

                    input_tok = tokenizer.decode([input_ids[i].item()])
                    target_tok = tokenizer.decode([target_ids[i].item()])

                    # Get logits for this position
                    logits = outputs[i]
                    top5_ids = logits.topk(5).indices.tolist()
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]

                    # Check if target is in top-5
                    status = "✅" if target_ids[i].item() in top5_ids else "❌"

                    print(f"{input_tok:>12} → {target_tok:<12} | Top-5: {top5_tokens} {status}")

            loss = loss_fn(outputs, target.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            pred = outputs.argmax(dim=1)  # (B*T,)
            target_flat = target.view(-1)  # (B*T,)
            correct = (pred == target_flat).sum().item()
            total_correct += correct
            total_tokens += target_flat.numel()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_train_loss:.4f}',
            })

        train_acc = total_correct / total_tokens

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_tokens = 0
            total_correct = 0
            total_topk_correct = 0
            total_perplexity = 0
            k = 5  # for top-k accuracy
            for batch_idx, (inputs, target) in enumerate(val_loader):
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, target.view(-1))
                total_val_loss += loss.item()

                # Calculate accuracy correctly
                pred = outputs.argmax(dim=1)  # (B*T,)
                target_flat = target.view(-1)  # (B*T,)

                # Top-1 accuracy
                total_correct += (pred == target_flat).sum().item()
                total_tokens += target_flat.numel()

                # Top-k accuracy
                top_k_preds = outputs.topk(k, dim=1).indices
                topk_correct = top_k_preds.eq(target_flat.unsqueeze(1)).any(dim=1).sum().item()
                total_topk_correct += topk_correct

                # Perplexity (sum of exp(loss) for averaging)
                total_perplexity += torch.exp(loss).item()

        # Averages
        val_acc = total_correct / total_tokens
        val_topk_acc = total_topk_correct / total_tokens
        avg_val_loss = total_val_loss / len(val_loader)
        avg_perplexity = total_perplexity / len(val_loader)
        avg_train_loss = total_train_loss / len(train_loader)
        metrics_logger.log_epoch(avg_train_loss, avg_val_loss, train_acc, val_acc, val_topk_acc, avg_perplexity)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, "
            f"Train Acc = {train_acc:.2%}, "
            f"Val Acc = {val_acc:.2%}, "
            f"Top-{k} Acc = {val_topk_acc:.2%}, "
            f"PPL = {avg_perplexity:.2f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f'New best validation loss: {best_val_loss:.4f}')
            save_checkpoint(
                model, optimizer, epoch,
                metrics_logger.get_metrics_history(),
                os.path.join(config['output_dir'], 'best_model.pth')
            )
            print(os.path.join(config['output_dir'], 'best_model.pth'))
            # logger.log({"best_val_loss": best_val_loss})

        # Plot and save metrics
        plot_metrics(metrics_logger.get_metrics_history(), config['output_dir'])

# Testing
    prompt = "The economy is"
    tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokens['input_ids'][:, :-1].to(device)  # remove last token if needed
    model.eval()
    with torch.no_grad():
        output = model(input_ids)

    # Get last position's logits (before softmax)
    last_token_logits = output[-1]  # shape: (vocab_size,)
    predicted_token_id = last_token_logits.argmax(dim=-1).item()
    predicted_word = tokenizer.decode([predicted_token_id])
    print(f"Input: {prompt}")
    print(f"Predicted next word: {predicted_word}")

if __name__ == '__main__':
    train_model('config.yaml')