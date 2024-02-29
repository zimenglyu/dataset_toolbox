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



# label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_34_36_38_{}_label.csv".format(label)
# result_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_Flight_343638_{}/flight".format(label)

# label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_34_36_38_labelled.csv"
# result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_343638"

# label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_labelled.csv"
# result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_38"

label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_all_labelled.csv"
result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_343638_all"
# result_path = "/Users/zimenglyu/Documents/code/git/susi/testing_result_38_all"
divide = " & "


testing = True
labels=[20,40,40,30,30,30]
som_size_list = [5,7,10,15,20,25]
neighbor_list = [3,3,3,5,3,3]
norm_list = ["standard", "standard", "standard",  "minmax", "standard", "standard"]
for som_size, norm, neighbor, label in zip(som_size_list, norm_list, neighbor_list, labels):
    mean_acc = []
    if testing:
        row_string = "& Test"
        filename = "{}/label_{}/{}/flight_{}_20000_vote_neighbor_{}_{}_0.csv".format(result_path, label, som_size, som_size, neighbor, norm)
    else:
        row_string = "& Validate"
        # result_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_Flight_38/SOM_Flight_{}/flight".format(label)
        # label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_{}_label.csv".format(label)
        label_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_34_36_38_{}_label.csv".format(label)
        result_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_Flight_343638_{}/flight".format(label)
        filename = "{}/{}/flight_{}_20000_vote_neighbor_{}_{}_0.csv".format(result_path, som_size, som_size, neighbor, norm)
    df = pd.read_csv(filename)
    true_labels = pd.read_csv(label_path)["FlPhase"].to_numpy()
    predicted_labels = df["FlPhase"].to_numpy().astype(int)
    per_class_accuracy = calculate_per_class_accuracy(true_labels, predicted_labels, [0,1,2,3,4,5])
    for cls, acc in per_class_accuracy.items():
        row_string += divide + f"{acc * 100:.2f}" 
        mean_acc.append(acc*100)
    row_string += divide + "{:.2f}".format(np.mean(mean_acc))
    row_string += " \\\ "
    print(row_string)
