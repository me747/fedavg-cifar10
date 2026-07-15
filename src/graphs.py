import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# defining a basic plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# file paths for training and validation metrics
train_path = "../results/training_logs.csv"
round_path = "../results/round_metrics.csv"

# ensure results folder exists
os.makedirs("../results", exist_ok=True)

train_data = pd.read_csv(train_path)
round_data = pd.read_csv(round_path)

# get average training loss and accuracy for each round
train_mean = train_data.groupby("round")[["train_loss", "train_acc"]].mean().reset_index()
train_mean["train_acc"] *= 100  # convert from fraction to percentage

# Loss vs Communication Round
plt.plot(train_mean["round"], train_mean["train_loss"], 'o-', label="Train Loss")
plt.plot(round_data["round"], round_data["val_loss"], 'o-', label="Validation Loss")
plt.xlabel("Communication Round")  # x-axis: rounds
plt.ylabel("Loss")                  # y-axis: loss
plt.title("Train vs Validation Loss")
plt.legend()
plt.savefig("../results/loss_curve.png", dpi=300)
plt.close()

# Accuracy vs Communication Round
plt.plot(train_mean["round"], train_mean["train_acc"], 'o-', label="Train Accuracy")
plt.plot(round_data["round"], round_data["val_acc"], 'o-', label="Validation Accuracy")
plt.xlabel("Communication Round")  # x-axis: rounds
plt.ylabel("Accuracy (%)")         # y-axis: accuracy in percentage
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.savefig("../results/accuracy_curve.png", dpi=300)
plt.close()

print("Figures saved inside results folder")
