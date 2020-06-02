import numpy as np
from matplotlib import pyplot as plt

from SDCA.sdca4crf.parameters.sparse_weights import SparsePrimalDirection
from SDCA.sdca4crf.parameters.weights import WeightsWithoutEmission
from SDCA.sdca4crf.utils import letters2wordimage


class DenseWeights(WeightsWithoutEmission):
    """Implement the weights of the model.

    Support all the operations necessary for the CRF and the optimization.
    Is also used to store the primal direction for dense data.
    """

    def __init__(self, emission=None, bias=None, transition=None,
                 nb_labels=0, nb_features=0, is_dataset_sparse=False):

        super().__init__(bias=bias, transition=transition, nb_labels=nb_labels)

        self.is_dataset_sparse = is_dataset_sparse
        self.emission = np.zeros([nb_labels, nb_features]) if emission is None else emission

    # BUILD THE WEIGHTS FROM DATA
    def add_datapoint(self, points_sequence, labels_sequence):
        super().add_datapoint(points_sequence, labels_sequence)

        if self.is_dataset_sparse:
            for point, label in zip(points_sequence, labels_sequence):
                self.emission[label, point[point >= 0]] += 1
        else:
            for point, label in zip(points_sequence, labels_sequence):
                self.emission[label] += point

    def add_centroid(self, points_sequence, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        super().add_centroid(points_sequence, marginals)
        if self.is_dataset_sparse:  # slow?
            for point, unimarginal in zip(points_sequence, marginals.unary):
                self.emission[:, point[point >= 0]] += unimarginal[:, np.newaxis]
        else:
            self.emission += np.dot(marginals.unary.T, points_sequence)

    @classmethod
    def from_marginals(cls, points_sequence, marginals):
        """Initialize the primal direction."""
        # This being called means that the data set is dense
        weights = cls(nb_features=points_sequence.shape[1], nb_labels=marginals.nb_labels,
                      is_dataset_sparse=False)
        weights.add_centroid(points_sequence, marginals)
        return weights

    # USE THE MODEL ON DATA
    def scores(self, points_sequence):
        unary_scores, binary_scores = super().scores(points_sequence)

        if self.is_dataset_sparse:  # slow?
            for t, point in enumerate(points_sequence):
                unary_scores[t] += self.emission[:, point[point >= 0]].sum(axis=1)
        else:
            unary_scores += np.dot(points_sequence, self.emission.T)

        return unary_scores, binary_scores

    # ARITHMETIC OPERATIONS
    def __mul__(self, scalar):
        tmp = super().__mul__(scalar)
        emission = scalar * self.emission
        return DenseWeights(emission, tmp.bias, tmp.transition,
                            is_dataset_sparse=self.is_dataset_sparse)

    def __iadd__(self, other):
        tmp = super().__iadd__(other)
        if isinstance(other, SparsePrimalDirection):
            self.emission[:, other.sparse_emission.active_set] += other.sparse_emission.values
        else:
            self.emission += other.emission
        return DenseWeights(self.emission, tmp.bias, tmp.transition,
                            is_dataset_sparse=self.is_dataset_sparse)

    def __sub__(self, other):
        tmp = super().__sub__(other)
        emission = self.emission - other.emission
        return DenseWeights(emission, tmp.bias, tmp.transition,
                            is_dataset_sparse=self.is_dataset_sparse)

    def squared_norm(self):
        return super().squared_norm() + np.sum(self.emission ** 2)

    def inner_product(self, other):
        ans = super().inner_product(other)
        if isinstance(other, SparsePrimalDirection):
            return ans + np.sum(self.emission[:, other.sparse_emission.active_set] *
                                other.sparse_emission.values)
        else:
            return ans + np.sum(self.emission * other.emission)

    # MISCELLANEOUS
    def display(self):
        super().display()
        if self.is_dataset_sparse:
            emissions = letters2wordimage(self.emission)
            plt.matshow(emissions, cmap="Greys")
            ticks_positions = np.linspace(0, emissions.shape[1],
                                          self.emission.shape[0] + 2).astype(int)[1:-1]
            plt.xticks(ticks_positions, np.arange(self.emission.shape[0]))
            plt.colorbar(fraction=0.046, pad=0.04)

    def to_array(self):
        return [self.transition, self.bias, self.emission]
