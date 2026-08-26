import matplotlib.pyplot as plt

train_losses = [
    9.15,
    7.36,
    6.69
]

val_losses = [
    9.12,
    7.30,
    6.81
]

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Evaluation Step")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.show()