import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

vanilla_train_path = "/home/brenta/scratch/jason/logs/imagenet/vanilla/exp_10/log_1272020_152250.csv"

def log_csv_to_numpy_array(log_csv_path):

    lines = open(log_csv_path, 'r').readlines()
    minibatches = []
    train_accs = []
    val_accs = []

    for line in lines[1:]:
        parts = line[:-1].split(",")
        minibatch = int(parts[1])
        train_acc = float(parts[3])
        val_acc = float(parts[5])
        minibatches.append(minibatch)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    return minibatches, train_accs, val_accs 

def plot_vanilla_examples():

    vanilla_output_path = "plots/imagenet/vanilla_training_exp_10.png"
    minibatches, train_accs, val_accs = log_csv_to_numpy_array(vanilla_train_path)
    fig, ax = plt.subplots()
    plt.ylim([-0.02, 1.02])
    plt.plot(minibatches, train_accs, label="Training Accuracy")
    plt.plot(minibatches, val_accs, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Vanilla performance on predicting image rotations (ResNet18, ImageNet)")
    plt.xlabel("Minibatch Number (128 images per minibatch)")
    plt.ylabel("Accuracy")
    plt.savefig(vanilla_output_path, dpi=400)

if __name__ == "__main__":

    plot_vanilla_examples()