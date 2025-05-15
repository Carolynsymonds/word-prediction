import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_dataloader
from model import TextGen
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from metrics import MetricsLogger
from utils import plot_metrics, load_config

def train(model, epochs, train_dataloader, val_dataloader, criterion, vocab_size):
    model.train()
    total_tokens = 0
    correct_tokens = 0
    correct_top5_tokens = 0
    best_val_loss = float('inf')
    metrics_logger = MetricsLogger()

    for epoch in range(epochs):
        running_loss = 0
        for input_seq, target_seq in train_dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, vocab_size)

            loss = criterion(outputs, target_seq.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().numpy()

            # Metrics
            # Compute accuracy
            predicted = outputs.argmax(dim=-1)
            correct = (predicted == target_seq).sum().item()
            correct_tokens += correct

            # Top-5 accuracy
            top5 = outputs.topk(5, dim=-1).indices
            target_expanded = target_seq.unsqueeze(1).expand_as(top5)
            top5_correct = (top5 == target_expanded).any(dim=1).sum().item()
            correct_top5_tokens += top5_correct

            total_tokens += target_seq.numel()

        train_epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_tokens / total_tokens
        top5_accuracy = correct_top5_tokens / total_tokens
        print(f"Train - Epoch {epoch} loss: {train_epoch_loss:.3f} epoch_accuracy: {epoch_accuracy:.3f}  top5_accuracy: {top5_accuracy:.3f}")
        # ---- Validation ----
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for input_seq, target_seq in val_dataloader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                outputs = model(input_seq)
                target_seq = target_seq.contiguous().view(-1)
                outputs = outputs.view(-1, vocab_size)

                loss = criterion(outputs, target_seq.view(-1))
                val_loss += loss.detach().cpu().numpy()

                # Accuracy
                predicted = outputs.argmax(dim=-1)
                correct_tokens += (predicted == target_seq).sum().item()
                total_tokens += target_seq.numel()

        val_epoch_loss = val_loss / len(val_dataloader)
        val_accuracy = correct_tokens / total_tokens
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_epoch_loss:.4f}, "
            f"Val Loss = {val_epoch_loss:.4f}, "
            f"Train Acc = {epoch_accuracy:.2%}, "
            f"Val Acc = {val_accuracy:.2%}, "
            f"Top-5 Acc = {top5_accuracy:.2%}, "
        )
        metrics_logger.log_epoch(train_epoch_loss, val_epoch_loss, epoch_accuracy, val_accuracy, top5_accuracy)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            print(f'New best validation loss: {best_val_loss:.4f}')
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "best_model.pt")
            print(f"Model saved to best_model.pt")

    return metrics_logger
if __name__ == "__main__":
    config = load_config('config.yaml')

    sequence_length = config['sequence_length']
    batch_size = config['batch_size']
    epochs = config['num_epochs']
    learning_rate = config['learning_rate']

    train_dataloader, val_dataloader, vocab_size, int_to_word = load_dataloader(sequence_length, batch_size)

    model = TextGen(
        vocab_size=vocab_size,
        embed_dim=100,
        num_layers=2,
        num_heads=2,
        sequence_length=sequence_length
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(model)

    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    # TRAIN
    metrics_logger = train(model, epochs, train_dataloader, val_dataloader, criterion, vocab_size)

    plot_metrics(metrics_logger.get_metrics_history(), config['output_dir'])





