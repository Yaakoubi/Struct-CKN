import numpy as np

from SDCA.sdca4crf.parameters.weights import WeightsWithoutEmission


class SparsePrimalDirection(WeightsWithoutEmission):

    def __init__(self, sparse_emission=None, bias=None, transition=None,
                 nb_labels=0):
        super().__init__(bias, transition, nb_labels)
        self.sparse_emission = sparse_emission

    def __mul__(self, scalar):
        tmp = super().__mul__(scalar)
        return SparsePrimalDirection(self.sparse_emission * scalar, tmp.bias, tmp.transition)

    @classmethod
    def from_marginals(cls, points_sequence, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        sparse_emission = SparseEmission.from_marginals(points_sequence, marginals)
        tmp = super(SparsePrimalDirection, cls).from_marginals(points_sequence, marginals)
        return cls(sparse_emission, tmp.bias, tmp.transition)

    def squared_norm(self):
        ans = super().squared_norm()
        return ans + self.sparse_emission.squared_norm()


class SparseEmission:

    def __init__(self, active_set, values):
        self.active_set = active_set
        self.values = values

    @classmethod
    def from_marginals(cls, points_sequence, marginals):
        alphalen = marginals.nb_labels

        active_set, inverse = np.unique(points_sequence, return_inverse=True)
        centroid = np.zeros([active_set.shape[0], alphalen])
        inverse = inverse.reshape(points_sequence.shape)
        for inv, marg in zip(inverse, marginals.unary):
            centroid[inv] += marg

        # Finally remove the absent attributes
        if active_set[0] == -1:
            active_set = active_set[1:]
            centroid = centroid[1:]
        else:
            pass

        centroid = np.transpose(centroid)
        return cls(active_set, centroid)

    def __mul__(self, scalar):
        return SparseEmission(self.active_set, scalar * self.values)

    def squared_norm(self):
        return np.sum(self.values ** 2)
