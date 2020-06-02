import numpy as np

from SDCA.sdca4crf.oracles import sequence_sum_product
from SDCA.sdca4crf.parameters.dense_weights import DenseWeights
from SDCA.sdca4crf.parameters.sequence_marginals import SequenceMarginals
from SDCA.sdca4crf.parameters.sparse_weights import SparsePrimalDirection


def initialize(warm_start, data, regularization):
    if isinstance(warm_start, np.ndarray):
        # assume that init contains the marginals for a warm start.
        if warm_start.shape[0] != len(data):
            raise ValueError(
                "Not the same number of warm start marginals (%i) and data points (%i)."
                % (warm_start.shape[0], len(data)))
        marginals = warm_start

    else:  # empirical initialization
        # The empirical marginals give a good value of the dual objective : 0,
        # and primal objective : average sequence length times log alphabet-size = 23
        # but the entropy has an infinite slope and curvature in the corners
        # of the simplex. Hence we take a convex combination between a lot of
        # empirical and a bit of uniform.
        # This is the recommended initialization for online exponentiated
        # gradient in appendix D of the SAG-NUS for CRF paper
        marginals = []
        for _, labels_sequence in data:
            marginals.append(dirac(labels_sequence, data.nb_labels))
        marginals = np.array(marginals)

    # Initialize the weights as the centroid of the ground truth features minus the centroid
    # of the features given by the marginals.
    ground_truth_centroid = centroid(data)
    marginals_centroid = centroid(data, marginals)
    weights = ground_truth_centroid - marginals_centroid
    weights *= 1 / regularization

    return marginals, weights, ground_truth_centroid


def centroid(data, marginals=None):
    ans = DenseWeights(nb_features=data.nb_features, nb_labels=data.nb_labels,
                       is_dataset_sparse=data.is_sparse)

    if marginals is None:  # ground truth centroid
        for point, label in data:
            ans.add_datapoint(point, label)
    else:  # marginals centroid
        for (point, _), margs in zip(data, marginals):
            ans.add_centroid(point, margs)

    ans *= 1 / len(data)

    return ans


def compute_primal_direction(points_sequence, dual_direction,
                             is_sparse, nb_samples, regularization):
    primal_direction_cls = SparsePrimalDirection if is_sparse else DenseWeights
    primal_direction = primal_direction_cls.from_marginals(points_sequence, dual_direction)

    # Centroid of the corrected features in the dual direction
    # = Centroid of the real features in the opposite of the dual direction
    primal_direction *= -1 / regularization / nb_samples

    return primal_direction


def uniform(length, nb_class, log=True):
    """Return uniform marginals for a sequence.

    :param length: of the sequence
    :param nb_class: number of possible labels for each variable in the sequence
    :param log: if true return the log-marginals
    """
    unary = np.ones([length, nb_class])
    binary = np.ones([length - 1, nb_class, nb_class])
    if log:
        unary *= - np.log(nb_class)
        binary *= -2 * np.log(nb_class)
    else:
        unary /= nb_class
        binary /= nb_class ** 2
    return SequenceMarginals(unary=unary, binary=binary, log=log)


def dirac(labels_sequence, nb_class, log=True):
    """Return smoothed dirac marginals over the observed sequence of labels.

    :param labels_sequence:
    :param nb_class:
    :param log: if True, return smoothed log-probabilities
    """
    length = len(labels_sequence)
    constant = 10 if log else 1

    unary_scores = np.zeros([length, nb_class])
    unary_scores[np.arange(length), labels_sequence] = constant
    binary_scores = np.zeros([length - 1, nb_class, nb_class])
    binary_scores[np.arange(length - 1), labels_sequence[:-1], labels_sequence[1:]] = constant

    if log:
        unary_marginal, binary_marginal, _ = sequence_sum_product(
            unary_scores, binary_scores)
    else:
        unary_marginal, binary_marginal = unary_scores, binary_scores
    return SequenceMarginals(unary=unary_marginal, binary=binary_marginal, log=log)
