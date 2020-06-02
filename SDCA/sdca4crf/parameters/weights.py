import matplotlib.pyplot as plt
import numpy as np

from SDCA.sdca4crf.oracles import sequence_sum_product, sequence_viterbi
from SDCA.sdca4crf.parameters.sequence_marginals import SequenceMarginals


class WeightsWithoutEmission:
    """Base class for the weights of the CRF. Include bias and transition weights.
    The emission weights are dealt with in the subclasses DenseWeights and SparseWeights.
    """

    def __init__(self, bias=None, transition=None, nb_labels=0):
        self.bias = np.zeros([nb_labels, 3]) if bias is None else bias
        self.transition = np.zeros([nb_labels, nb_labels]) if transition is None else transition
        self.nb_labels = self.transition.shape[0]

    # BUILD THE WEIGHTS FROM DATA
    def add_datapoint(self, points_sequence, labels_sequence):
        for t, label in enumerate(labels_sequence):
            self.bias[label] += [1, t == 0, t == len(labels_sequence) - 1]
        for t in range(labels_sequence.shape[0] - 1):
            self.transition[labels_sequence[t], labels_sequence[t + 1]] += 1

    def add_centroid(self, points_sequence, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        self.bias[:, 0] += np.sum(marginals.unary, axis=0)
        self.bias[:, 1] += marginals.unary[0]
        self.bias[:, 2] += marginals.unary[-1]

        self.transition += np.sum(marginals.binary, axis=0)

    @classmethod
    def from_marginals(cls, points_sequence, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        ans = cls(nb_labels=marginals.nb_labels)
        ans.add_centroid(points_sequence, marginals)
        return ans

    # USE THE MODEL ON DATA
    def scores(self, points_sequence):
        seq_len = points_sequence.shape[0]

        unary_scores = np.zeros([seq_len, self.nb_labels])
        unary_scores += self.bias[:, 0]  # model bias
        unary_scores[0] += self.bias[:, 1]  # beginning of word bias
        unary_scores[-1] += self.bias[:, 2]  # end of word bias

        binary_scores = (seq_len - 1) * [self.transition]

        return unary_scores, binary_scores

    def infer_probabilities(self, points_sequence):
        uscores, bscores = self.scores(points_sequence)
        umargs, bmargs, log_partition = sequence_sum_product(uscores, bscores)
        umargs = np.minimum(umargs, 0)
        bmargs = np.minimum(bmargs, 0)
        ans = SequenceMarginals(umargs, bmargs, log=True)

        return ans, log_partition

    def predict(self, points_sequence):
        uscores, bscores = self.scores(points_sequence)
        return sequence_viterbi(uscores, bscores)[0]

    def labeled_sequence_score(self, points_sequence, labels_sequence):
        """Return the score <self,F(points_sequence, labels_sequence)>."""
        ans = np.sum(self.bias[labels_sequence, 0])
        ans += self.bias[labels_sequence[0], 1]
        ans += self.bias[labels_sequence[-1], 2]

        ans += np.sum(self.transition[labels_sequence[:-1], labels_sequence[1:]])

        return ans

    # ARITHMETIC OPERATIONS
    def __mul__(self, scalar):
        bias = scalar * self.bias
        transition = scalar * self.transition
        return WeightsWithoutEmission(bias, transition)

    def __iadd__(self, other):
        self.bias += other.bias
        self.transition += other.transition
        return WeightsWithoutEmission(self.bias, self.transition)

    def __sub__(self, other):
        return WeightsWithoutEmission(
            bias=self.bias - other.bias,
            transition=self.transition - other.transition)

    def squared_norm(self):
        return np.sum(self.bias ** 2) + np.sum(self.transition ** 2)

    def inner_product(self, other):
        return np.sum(self.bias * other.bias) \
               + np.sum(self.transition * other.transition)

    # MISCELLANEOUS
    def display(self):
        """Display bias and transition features as heatmaps."""
        cmap = "Greys"
        plt.matshow(self.transition, cmap=cmap)
        plt.grid()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features", y=1.3)

        rescale_bias = np.array([1 / 23, 1, 1])
        plt.matshow((self.bias * rescale_bias).T, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features", y=1.15)

        plt.show()
