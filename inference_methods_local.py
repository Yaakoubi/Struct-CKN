import ad3
import numpy as np
from pystruct.inference.common import _validate_params


class InferenceException(Exception):
    pass


def inference_ad3_local(unary_potentials, pairwise_potentials, edges, relaxed=False,
                        verbose=0, return_energy=False, branch_and_bound=False,
                        inference_exception=None, yassine=False):

    b_multi_type = isinstance(unary_potentials, list)
    if b_multi_type:
        res = ad3.general_graph(unary_potentials, edges, pairwise_potentials,
                                verbose=verbose, n_iterations=4000, exact=branch_and_bound)
    else:
        n_states, pairwise_potentials = \
            _validate_params(unary_potentials, pairwise_potentials, edges)
        unaries = unary_potentials.reshape(-1, n_states)
        res = ad3.general_graph(unaries, edges, pairwise_potentials,
                                verbose=verbose, n_iterations=4000, exact=branch_and_bound)

    unary_marginals, pairwise_marginals, energy, solver_status = res
    if verbose:
        print(solver_status)

    if solver_status in ["fractional", "unsolved"] and relaxed:
        if b_multi_type:
            y = (unary_marginals, pairwise_marginals)
            if yassine and verbose:
                print("I'm here 1")
        else:
            unary_marginals = unary_marginals.reshape(unary_potentials.shape)
            y = (unary_marginals, pairwise_marginals)
            if yassine and verbose:
                print("I'm here 2")
    else:
        if b_multi_type:
            if inference_exception and solver_status in ["fractional", "unsolved"]:
                raise InferenceException(solver_status)
            ly = list()
            _cum_n_states = 0
            for unary_marg in unary_marginals:
                ly.append(_cum_n_states + np.argmax(unary_marg, axis=-1))
                _cum_n_states += unary_marg.shape[1]
            y = np.hstack(ly)
            if yassine and verbose:
                print("I'm here 3")
        else:
            y = np.argmax(unary_marginals, axis=-1)
            if yassine and verbose:
                print("I'm here 4")

    if return_energy:
        return y, -energy
    if yassine and verbose:
        print("I'm here 5")
    if yassine:
        return y, unary_marginals
    return y
