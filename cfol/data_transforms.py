import math
import numpy as np


def get_class_i(x, y, i, n_reduce=None):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    if n_reduce is not None:
        pos_i = pos_i[:n_reduce]
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i
    

def reduce_classes_dbset(data, targets, n_reduce=100, permute=True):
    """ Accepts a trainset torch db and reduces the samples for all classes in n_reduce. 
    """
    db1 = data
    lbls = targets
    n_classes = int(np.max(lbls)) + 1

    # # create the undersampled lists of data and samples.
    data, classes = [], []
    for cl_id in range(n_classes):
        samples = get_class_i(db1, lbls, cl_id, n_reduce=n_reduce)
        data.append(samples)
        classes.extend([cl_id] * len(samples))
        print(f'Datapoints for class {cl_id}: {len(samples)}')

    # # convert into numpy arrays.
    data, classes =  np.concatenate(data, axis=0), np.array(classes, dtype=np.int)
    if permute:
        # # optionally permute the data to avoid having them sorted.
        permut1 = np.random.permutation(len(classes))
        data, classes = data[permut1], classes[permut1]
    return data, classes


def imbalance_factor_to_lt_factor(imbalance_factor, db1, lbls):
    """
    Data assumed to have the same number of samples in all classes.

    imbalance factor = num_samples_largest_c/num_samples_smallest
    
        largest*lt_factor^(num_classes-1) = smallest 
    <=> lt_factor = (smallest/largest)^(1/(num_classes-1))
    """
    n_classes = int(np.max(lbls)) + 1
    num_largest = int(db1.shape[0] // n_classes)

    num_smallest = num_largest / imbalance_factor
    return (num_smallest/num_largest)**(1/(n_classes-1))


def reduce_classes_dbset_longtailed(db1, lbls, permute=True, lt_factor=None):
    """ Accepts a trainset torch db (which is assumed to have the same 
        number of samples in all classes) and creates a long-tailed 
        distribution with factor reduction factor.
        """
    n_classes = int(np.max(lbls)) + 1
    n_samples_class = int(db1.shape[0] // n_classes)
    # # create the undersampled lists of data and samples.
    data, classes = [], []
    for cl_id in range(n_classes):
        n_reduce = int(n_samples_class * lt_factor ** cl_id)
        samples = get_class_i(db1, lbls, cl_id, n_reduce=n_reduce)
        data.append(samples)
        classes.extend([cl_id] * n_reduce)
        print(f'Datapoints for class {cl_id}: {n_reduce}')
    # # convert into numpy arrays.
    data, classes =  np.concatenate(data, axis=0), np.array(classes, dtype=np.int)
    if permute:
        # # optionally permute the data to avoid having them sorted.
        permut1 = np.random.permutation(len(classes))
        data, classes = data[permut1], classes[permut1]
    return data, classes
