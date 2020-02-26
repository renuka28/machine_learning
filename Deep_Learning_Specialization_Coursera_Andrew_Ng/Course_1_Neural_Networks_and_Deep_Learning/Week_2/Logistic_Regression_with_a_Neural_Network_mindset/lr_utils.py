import numpy as np
import h5py
import os


def load_dataset():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    full_train_file_name = os.path.join(dir_name, "datasets")
    full_train_file_name = os.path.join(
        full_train_file_name, "train_catvnoncat.h5")
    # print(full_train_file_name)
    train_dataset = h5py.File(full_train_file_name, "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    full_test_file_name = os.path.join(dir_name, "datasets")
    full_test_file_name = os.path.join(
        full_test_file_name, "test_catvnoncat.h5")
    # print(full_test_file_name)
    test_dataset = h5py.File(full_test_file_name, "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
