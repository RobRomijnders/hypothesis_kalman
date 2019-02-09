import numpy as np
import matplotlib.pyplot as plt
from os.path import join


def load_data(direc):
    dataset = 'ECG5000'

    # Read the data
    datadir = join(direc, dataset) + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')
    data_total = np.concatenate((data_train, data_test_val), axis=0)

    data = data_total[:, 1:]
    labels = data_total[:, 0]

    # start with the labels from 0
    smallest_label = np.min(labels)
    if smallest_label > 0:
        labels -= smallest_label

    return data, labels


def get_dataset(direc, data_option, num_samples):
    assert data_option in [1, 2], "only hypothesis 1 and 2 are defined"
    data, labels = load_data(direc)

    data0 = data[labels==0]
    data1 = data[labels==1]

    if data_option == 1:
        random_indices = np.random.choice(len(data0), size=2*num_samples, replace=False)
        data_samples = data0[random_indices]
    else:
        random_indices_0 = np.random.choice(len(data0), size=num_samples, replace=False)
        random_indices_1 = np.random.choice(len(data1), size=num_samples, replace=False)
        data_samples = np.concatenate((data0[random_indices_0], data1[random_indices_1]), axis=0)

    return data_samples




def plot_data(data, labels):
    plot_row = 4

    unique_labels = np.unique(labels)
    num_classes = int(np.max(unique_labels) + 1)

    f, axarr = plt.subplots(plot_row, num_classes)
    for c in range(num_classes):    #Loops over classes, plot as columns
        ind = np.where(labels == c)[0]
        if len(ind) == 0:
            continue

        ind_plot = np.random.choice(ind, size=plot_row)
        for num_row in range(plot_row):  #Loops over rows
            axarr[num_row, c].plot(data[ind_plot[num_row],:])
            # Only shops axes for bottom row and left column
            if not num_row == plot_row-1:
                plt.setp([axarr[num_row,c].get_xticklabels()], visible=False)
            if not c == 0:
                plt.setp([axarr[num_row,c].get_yticklabels()], visible=False)
            if num_row == 0:
                axarr[num_row, c].set_title(f'class {c}')
    f.subplots_adjust(hspace=0)  #No horizontal space between subplots
    f.subplots_adjust(wspace=0)  #No vertical space between subplots
    plt.show()