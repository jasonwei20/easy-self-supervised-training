import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

prediction_csv_1_epoch = "/home/brenta/scratch/jason/outputs/image_net/vanilla/resnet18_e0_mb40000_va0.80428.pt/train_preds.csv"

def prediction_csv_to_correct_incorrect_confidences(prediction_csv):
    
    correct_confidences = []
    incorrect_confidences = []
    
    lines = open(prediction_csv, "r").readlines()
    for line in lines[1:]:
        parts = line[:-1].split(",")
        gt = parts[0].split("/")[0]
        pred = parts[1]
        conf = float(parts[2])
        if gt == pred:
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)
    
    return correct_confidences, incorrect_confidences

def plot_histogram(prediction_csv, title):
    correct_confidences, incorrect_confidences = prediction_csv_to_correct_incorrect_confidences(prediction_csv)

    output_path = 'plots/' + prediction_csv.split(".")[0].split("/")[-1] + '.png'
    fig, ax = plt.subplots()

    bins = np.linspace(0, 1, 1000)
    ax.hist(correct_confidences, bins, alpha=0.5, label="Correct")
    ax.hist(incorrect_confidences, bins, alpha=0.5, label="Wrong!")
    plt.xlim([-0.02, 1.02])
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xlabel("Predicted confidence")
    plt.savefig(output_path, dpi=400)
    print("output at", output_path)

if __name__ == "__main__":
    plot_histogram(prediction_csv_1_epoch, "Predicted distribution after 1 epoch")
