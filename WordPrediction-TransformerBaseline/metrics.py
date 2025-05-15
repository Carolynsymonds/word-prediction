class MetricsLogger:
    def __init__(self):
        # Initialize metrics storage
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_topk_acc = []

    def log_epoch(self, train_loss, val_loss, train_acc, val_acc, val_topk_acc):
        # Store metrics for one epoch
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.train_acc.append(train_acc)
        self.val_topk_acc.append(val_topk_acc)

    def get_metrics_history(self):
        # Return complete metrics history
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'train_acc': self.train_acc,
            'val_topk_acc': self.val_topk_acc,
        }
