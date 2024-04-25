from matplotlib import pyplot as plt


def plot_train_history(rewards, total_loss):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(rewards, label="Rewards")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Loss")
    plt.grid(True)

    plt.show()