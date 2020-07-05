# Main Script for HW 1

import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")


# Set Up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy import io

mnist = io.loadmat("data/mnist_data.mat")
spam = io.loadmat("data/spam_data.mat")
cifar10 = io.loadmat("data/cifar10_data.mat")


# Data Partitioning

def split_data(data, labels, val_size):
    """Function to split data into train and val sets

    Args:
        data (np array): Data containing X values
        labels (np array): Data containing y values
        val_count (integer): Size of the validation dataset
    """
    # Shuffle index
    index = np.random.permutation(len(data))

    # Split into Datasets
    X_val = data[index][-val_size:]
    X_train = data[index][:-val_size]
    y_val = labels[index][-val_size:].ravel()
    y_train = labels[index][:-val_size].ravel()

    return X_train, X_val, y_train, y_val


mnist_X_train, mnist_X_val, mnist_y_train, mnist_y_val = split_data(mnist['training_data'], mnist['training_labels'], 10000)
spam_X_train, spam_X_val, spam_y_train, spam_y_val = split_data(spam['training_data'], spam['training_labels'], int(len(spam['training_data']) * 0.2))
cifar10_X_train, cifar10_X_val, cifar10_y_train, cifar10_y_val = split_data(cifar10['training_data'], cifar10['training_labels'], 5000)


# Support Vector Machine and Learning Curves

def plot_learning_curve(X_train_all, X_val_all, y_train_all, y_val_all, train_sizes, title):
    """Function to plot the error rate vs training examples for SVM Classifier

    Args:
        X_train (numpy array): Training Data Examples
        y_train (numpy array): Labels for training data
        X_val (numpy array): Validation Data Examples
        y_val (numpy array): Labels for validation data
        example_sizes (list integers): List of example sizes to be plotted
        title (string): Title of Graph
    """

    errors_df = pd.DataFrame(columns = ['train_size', 'train_acc', 'val_acc'])

    # Loop through example sizes and get the training and validation error
    for train_size in train_sizes:
        # Select Subset of Data
        X_train = X_train_all[:train_size]
        X_val = X_val_all[:train_size]
        y_train = y_train_all[:train_size]
        y_val = y_val_all[:train_size]

        # Initialize Model
        model = svm.SVC(kernel='linear')

        # Fit model
        print(f"Training {title} using {train_size} examples")
        model.fit(X_train, y_train)

        # Get Predictions 
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Get Accuracy Score for X_Train and X_Val
        errors = pd.DataFrame({
            'train_size': [train_size],
            'train_acc': [accuracy_score(y_train, train_pred)],
            'val_acc': [accuracy_score(y_val, val_pred)]
            })
        
        # Concatenate Dataframes
        errors_df = pd.concat([errors_df, errors])

    # Plot Learning Curve
    fig, ax = plt.subplots()

    errors_df.plot(x='train_size', y='train_acc',kind='line', ax=ax)
    errors_df.plot(x='train_size', y='val_acc',kind='line', color='red', ax=ax)

    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    # Save Figure
    plt.savefig('figs/' + title + '_learning_curve.png')


mnist_train_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
spam_train_sizes = [100, 200, 500, 1000, 2000, len(spam_X_train)]
cifar10_train_sizes = [100, 200, 500, 1000, 2000, 5000]

# Plot Learning Curves
plot_learning_curve(mnist_X_train, mnist_X_val, mnist_y_train, mnist_y_val, mnist_train_sizes, 'mnist')
plot_learning_curve(spam_X_train, spam_X_val, spam_y_train, spam_y_val, spam_train_sizes, 'spam')
plot_learning_curve(cifar10_X_train, cifar10_X_val, cifar10_y_train, cifar10_y_val, cifar10_train_sizes, 'cifar10')


