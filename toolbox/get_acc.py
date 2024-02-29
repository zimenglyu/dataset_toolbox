import pandas as pd 
import numpy as np
from glob import glob
from collections import defaultdict
def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculate the accuracy of a classification model.

    Parameters:
    true_labels (list): The true labels of the data.
    predicted_labels (list): The labels predicted by the model.

    Returns:
    float: The accuracy of the model.
    """

    # Ensure that the two lists are of the same length
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true labels and predicted labels must be the same.")
    # print("true_labels: ", true_labels)
    # print("predicted_labels: ", predicted_labels)
    # Count the number of correct predictions
    correct_predictions = sum(t == p for t, p in zip(true_labels, predicted_labels))

    # Calculate accuracy
    accuracy = correct_predictions / len(true_labels)

    return accuracy

def calculate_per_class_accuracy(true_labels, predicted_labels, classes):
    """
    Calculate the per-class accuracy of a classification model.

    Parameters:
    true_labels (list): The true labels of the data.
    predicted_labels (list): The labels predicted by the model.
    classes (list): The list of unique classes.

    Returns:
    dict: A dictionary with classes as keys and their accuracies as values.
    """

    # Initialize dictionaries to hold the count of correct predictions and total count per class
    correct_count = defaultdict(int)
    total_count = defaultdict(int)

    # Iterate over true and predicted labels to populate the dictionaries
    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            correct_count[true] += 1
        total_count[true] += 1

    # Calculate per-class accuracy
    accuracies = {cls: correct_count[cls] / total_count[cls] if total_count[cls] > 0 else 0 for cls in classes}

    return accuracies

def get_neighbors(label):
    if (label == 5):
        neighbor_list = [3, 5, 7]
    elif (label == 10):
        neighbor_list = [3, 5, 7, 10]
    elif (label == 20):
        neighbor_list = [3, 5, 7, 10, 15]
    elif (label == 30):
        neighbor_list = [3, 5, 7, 10, 15]
    elif (label == 40):
        neighbor_list = [3, 5, 7, 10, 15]
    else:
        raise ValueError("Invalid label.")
    return neighbor_list

def get_som_size(label):
    if (label == 5):
        som_size_list = [5, 7, 10]
    elif (label == 10):
        som_size_list = [5, 7, 10, 15]
    elif (label == 20):
        som_size_list = [5, 7, 10, 15, 20, 25]
    elif (label == 30):
        som_size_list = [5, 7, 10, 15, 20, 25]
    elif (label == 40):
        som_size_list = [5, 7, 10, 15, 20, 25]
    else:
        raise ValueError("Invalid label.")
    return som_size_list

label = 30
testing = True
divide = " & "
neighbor_list = get_neighbors(label)
som_size_list = get_som_size(label)


# label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_34_36_38_{}_label.csv".format(label)
# result_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_Flight_343638_{}/flight".format(label)

# label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_34_36_38_labelled.csv"
# result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_343638"

# label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_labelled.csv"
# result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_38"

label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_all_labelled.csv"
result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_343638_all"
# result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_38_all"

true_labels = pd.read_csv(label_path)["FlPhase"].to_numpy()
final_acc = []

for som_size in som_size_list:
    row_string = "{} x {}".format(som_size, som_size)
    for norm in ["minmax",  "standard", "robust"]:
        for neighbor in neighbor_list:
            if testing:
                filename = "{}/label_{}/{}/flight_{}_20000_vote_neighbor_{}_{}_".format(result_path, label, som_size, som_size, neighbor, norm)
            else:
                filename = "{}/{}/flight_{}_20000_vote_neighbor_{}_{}_".format(result_path, som_size, som_size, neighbor, norm)
            files = glob(filename + "*.csv")
            num_files = len(files)
            if num_files == 0:
                # print("No file found for {}".format(filename))
                row_string += divide + "-"
                continue
            for i in range(num_files):
                df = pd.read_csv(files[i])
                predicted_labels = df["FlPhase"].to_numpy().astype(int)
                acc = calculate_accuracy(true_labels, predicted_labels)
                if i == 0:
                    accs = np.array([acc])
                else:
                    accs = np.append(accs, acc)
            avg_acc = np.mean(accs) * 100
            row_string += divide + "{:.2f}".format(avg_acc)
            final_acc.append(avg_acc)
    row_string += " \\\ "
    print(row_string)
    #         print("som_size: {}, norm: {}, neighbor: {}, avg_acc: {:.2f}".format(som_size, norm, neighbor, avg_acc))
    #     print("")
    # print("---------------------------------------")
print("best acc of all: {:.2f}".format(np.max(final_acc)))