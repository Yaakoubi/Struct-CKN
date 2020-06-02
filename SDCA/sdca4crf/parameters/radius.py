import numpy as np

from SDCA.sdca4crf.parameters.dense_weights import DenseWeights


def radius(points_sequence, labels_sequence, data):
    """Return \max_y \|F(x_i,y_i) - F(x_i,y) \|.

    :param points_sequence: sequence of points
    :param labels_sequence: sequence of labels
    :param data: data set with the specifications nb_features, nb_labels, is_sparse
    """
    weights_cls = DenseWeights
    featuremap = weights_cls(nb_features=data.nb_features, nb_labels=data.nb_labels,
                             is_dataset_sparse=data.is_sparse)

    # ground truth feature
    featuremap.add_datapoint(points_sequence, labels_sequence)

    # the optimal labels_sequence is made of only one label
    # that is the least present in the true labels_sequence
    # First find the labels present in the sequence
    ulabels, ucounts = np.unique(labels_sequence, return_counts=True)
    # Second find the labels absent in the sequence
    diff_labels = np.setdiff1d(np.arange(data.nb_labels), labels_sequence)
    if len(diff_labels) > 0:  # if there are some take one
        optlabel = diff_labels[0]
    else:  # else find the label which appears the least
        optlabel = ulabels[np.argmin(ucounts)]

    # Finally create a sequence with only this label
    optlabels_sequence = optlabel * np.ones_like(labels_sequence)
    # Add it to the featuremap
    featuremap *= -1
    featuremap.add_datapoint(points_sequence, optlabels_sequence)

    return np.sqrt(featuremap.squared_norm())


def radii(data):
    """Return an array with the radius of the corrected features for the input data points."""
    rs = np.empty(len(data))
    for i, (points_sequence, labels_sequence) in enumerate(data):
        rs[i] = radius(points_sequence, labels_sequence, data)
    return rs
