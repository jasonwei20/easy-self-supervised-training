import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

vanilla_train_path = "logs/log_1122020_223225.csv"

def log_csv_to_numpy_array(log_csv_path):

    lines = open(log_csv_path, 'r').readlines()
    epochs = []
    train_accs = []
    val_accs = []

    for line in lines[1:]:
        parts = line[:-1].split(",")
        epoch = int(parts[0])
        train_acc = float(parts[2])
        val_acc = float(parts[4])
        epochs.append(epoch)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    return epochs, train_accs, val_accs 

def plot_vanilla_examples():

    vanilla_output_path = "plots/vanilla_training.png"
    epochs, train_accs, val_accs = log_csv_to_numpy_array(vanilla_train_path)
    fig, ax = plt.subplots()
    plt.ylim([-0.02, 1.02])
    plt.plot(epochs, train_accs, label="Training Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Vanilla performance on predicting image rotations (ResNet18, VOC07)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(vanilla_output_path, dpi=400)

def print_distribution():

if __name__ == "__main__":

    plot_vanilla_examples()