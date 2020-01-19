import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

vanilla_train_path = "logs/vanilla/log_1122020_223225.csv"
ten_paths = [   "logs/start_at_10/easy10.csv", "logs/start_at_10/easy25.csv", "logs/start_at_10/hard10.csv", 
                "logs/start_at_10/hard25.csv", "logs/start_at_10/correct.csv", "logs/start_at_10/incorrect.csv"]
ten_path_to_legend = {"logs/start_at_10/easy10.csv": "Easiest 10%", "logs/start_at_10/easy25.csv": "Easiest 25%", "logs/start_at_10/hard10.csv": "Hardest 10%", 
                "logs/start_at_10/hard25.csv": "Hardest 25%", "logs/start_at_10/correct.csv": "Correct data", "logs/start_at_10/incorrect.csv": "Incorrect data"}
        

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

def plot_starting_from_ten():

    num_epochs = 60
    ten_start_output_path = "plots/ten_start.png"
    epochs, train_accs, val_accs = log_csv_to_numpy_array(vanilla_train_path)
    fig, ax = plt.subplots()
    plt.ylim([-0.02, 1.02])
    plt.plot(epochs[:num_epochs], val_accs[:num_epochs], label="Vanilla")

    for ten_path in ten_paths:
        epochs, train_accs, val_accs = log_csv_to_numpy_array(ten_path)
        plt.plot(epochs[:num_epochs-11], val_accs[:num_epochs-11], label=ten_path_to_legend[ten_path] )

    plt.legend(loc="lower right")
    plt.title("CL performance on predicting image rotations (ResNet18, VOC07)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(ten_start_output_path, dpi=400)
    

if __name__ == "__main__":

    # plot_vanilla_examples()
    plot_starting_from_ten()