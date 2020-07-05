# Main Script for HW 1

import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")


# Set Up
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io

for data_name in ["mnist", "spam", "cifar10"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)


mnist = io.loadmat("data/mnist_data.mat")
spam = io.loadmat("data/spam_data.mat")
cifar10 = io.loadmat("data/cifar10_data.mat")

# Data Partitioning

def split_data(data, labels, val_count):
    """Function to split data into train and val sets

    Args:
        data (np array): Data containing X values
        labels (np array): Data containing y values
        val_count (integer): Size of the validation dataset
    """
    # Shuffle index
    index = np.random.permutation(len(data))

    # Split into Datasets
    X_val = data[index][-val_count:]
    X_train = data[index][:val_count]
    y_val = labels[index][-val_count:]
    y_train = labels[index][:val_count]

    return X_train, X_val, y_train, y_val


mnist_X_train, mnist_X_val, mnist_y_train, mnist_y_val = split_data(mnist['training_data'], mnist['training_labels'], 10000)
spam_X_train, spam_X_val, spam_y_train, spam_y_val = split_data(spam['training_data'], spam['training_labels'], int(len(spam['training_data']) * 0.2))
cifar10_X_train, cifar10_X_val, cifar10_y_train, cifar10_y_val = split_data(cifar10['training_data'], cifar10['training_labels'], 5000)

print(f"Rows in mnist_val is {len(mnist_X_val)}")
print(f"Rows in spam_val is {len(spam_X_val)}")
print(f"Rows in cifar10_val is {len(cifar10_X_val)}")


# Support Vector Machine

