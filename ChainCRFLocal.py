import numpy as np
from pystruct.inference.maxprod import inference_max_product
from pystruct.models import ChainCRF

from inference_methods_local import inference_ad3_local


def make_chain_edges(x):
    inds = np.arange(x.shape[0])
    edges = np.concatenate([inds[:-1, np.newaxis], inds[1:, np.newaxis]],
                           axis=1)
    return edges


class ChainCRFLocal(ChainCRF):
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 class_weight=None, directed=True):
        ChainCRF.__init__(self, n_states, n_features, inference_method,
                          class_weight, directed)

    def turn_dense_to_one_matrix(self, a, m):
        n = len(a)
        b = np.ones((n, m)) * 1e-4
        b[np.arange(n), a] = 1
        b = b / b.sum(axis=1, keepdims=1)
        return b

    def batch_marginals(self, param_x, param_y, w, num_classes=None):
        ys = []
        marginalss = []
        for x in param_x:
            if self.inference_method == "max-product":
                y = inference_max_product(self._get_unary_potentials(
                    x, w), self._get_pairwise_potentials(x, w), self._get_edges(x), relaxed=True)
                marginals = self.turn_dense_to_one_matrix(y, num_classes)
            elif self.inference_method == "ad3":
                y, marginals = inference_ad3_local(self._get_unary_potentials(x, w), self._get_pairwise_potentials(
                    x, w), self._get_edges(x), relaxed=True, branch_and_bound=False, return_marginals=True)
            else:
                print("Unknown inference method !")
            ys.append(y)
            marginalss.append(marginals)
        return np.array(ys), np.array(marginalss)

    def marginals(self, x, y, w):
        pairwise_potentiels = self._get_pairwise_potentials(x, w)
        unary_potentials = self._get_unary_potentials(x, w)
        edges = self._get_edges(x)
        return inference_ad3_local(unary_potentials, pairwise_potentiels, edges, relaxed=True, branch_and_bound=False,
                                   return_marginals=True)
