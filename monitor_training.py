# monitor_training.py
import matplotlib.pyplot as plt


def plot_training_logs(log_file="logs/training.log"):
    """Визуализация логов обучения"""

    losses = []
    eval_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            if "loss" in line and "eval_loss" not in line:
                try:
                    # Ищем значение loss
                    parts = line.split("loss")
                    if len(parts) > 1:
                        loss_val = float(parts[1].split(":")[1].strip().split(",")[0])
                        losses.append(loss_val)
                except:
                    pass

            if "eval_loss" in line:
                try:
                    parts = line.split("eval_loss")
                    eval_loss = float(parts[1].split(":")[1].strip().split(",")[0])
                    eval_losses.append(eval_loss)
                except:
                    pass

    # График
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.plot(eval_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Прогресс обучения')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

    print(f"Минимальный loss: {min(losses) if losses else 'N/A'}")
    print(f"Минимальный eval_loss: {min(eval_losses) if eval_losses else 'N/A'}")