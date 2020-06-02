import numpy as np

from SDCA.sdca4crf.utils import logsumexp


def sequence_sum_product(uscores, bscores):
    """Apply the sum-product algorithm on a chain

    :param uscores: array T*K, (unary) scores on individual nodes
    :param bscores: array (T-1)*K*K, (binary) scores on the edges
    :return: log-marginals on nodes, log-marginals on edges, log-partition
    """

    # I keep track of the islog messages instead of the messages
    # This is more stable numerically

    length, nb_class = uscores.shape

    if length == 1:
        log_partition = logsumexp(uscores[0])
        umargs = uscores - log_partition
        bmargs = np.zeros([length - 1, nb_class, nb_class])
        return umargs, bmargs, log_partition

    bm = np.zeros([length - 1, nb_class])  # backward_messages
    fm = np.zeros([length - 1, nb_class])  # forward_messages

    # backward pass
    bm[-1] = logsumexp(bscores[-1] + uscores[-1], axis=-1)
    for t in range(length - 3, -1, -1):
        bm[t] = logsumexp(bscores[t] + uscores[t + 1] + bm[t + 1], axis=-1)

    # we compute the log-partition and include it in the forward messages
    log_partition = logsumexp(bm[0] + uscores[0])

    # forward pass
    fm[0] = logsumexp(bscores[0].T + uscores[0] - log_partition, axis=-1)
    for t in range(1, length - 1):
        fm[t] = logsumexp(bscores[t].T + uscores[t] + fm[t - 1], axis=-1)

    # unary marginals
    umargs = np.empty([length, nb_class])
    umargs[0] = uscores[0] + bm[0] - log_partition
    umargs[-1] = fm[-1] + uscores[-1]
    for t in range(1, length - 1):
        umargs[t] = fm[t - 1] + uscores[t] + bm[t]

    # binary marginals
    bmargs = np.empty([length - 1, nb_class, nb_class])

    if length == 2:
        bmargs[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] - log_partition
    else:
        bmargs[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] + bm[1] - log_partition
        bmargs[-1] = fm[-2, :, np.newaxis] + uscores[-2, :, np.newaxis] + bscores[-1] + uscores[-1]
        for t in range(1, length - 2):
            bmargs[t] = fm[t - 1, :, np.newaxis] + uscores[t, :, np.newaxis] + bscores[t] + \
                        uscores[t + 1] + bm[t + 1]

    return umargs, bmargs, log_partition


def sequence_viterbi(uscores, bscores):
    # I keep track of the score instead of the potentials
    # because summation is more stable than multiplication
    length, nb_class = uscores.shape
    if length == 1:
        global_argmax = np.argmax(uscores[0])
        return global_argmax, uscores[0, global_argmax]

    # backward pass
    argmax_messages = np.empty([length - 1, nb_class], dtype=int)
    max_messages = np.empty([length - 1, nb_class], dtype=float)
    tmp = bscores[-1] + uscores[-1]
    # Find the arg max
    argmax_messages[-1] = np.argmax(tmp, axis=-1)
    # Store the max
    max_messages[-1] = tmp[np.arange(nb_class), argmax_messages[-1]]
    for t in range(length - 3, -1, -1):
        tmp = bscores[t] + uscores[t + 1] + max_messages[t + 1]
        argmax_messages[t] = np.argmax(tmp, axis=-1)
        max_messages[t] = tmp[np.arange(nb_class), argmax_messages[t]]

    # Label to be returned
    global_argmax = np.empty(length, dtype=int)

    # forward pass
    tmp = max_messages[0] + uscores[0]
    global_argmax[0] = np.argmax(tmp)
    global_max = tmp[global_argmax[0]]
    for t in range(1, length):
        global_argmax[t] = argmax_messages[t - 1, global_argmax[t - 1]]

    return global_argmax, global_max
