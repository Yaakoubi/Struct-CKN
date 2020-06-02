# standard imports
import matplotlib.pyplot as plt
import numpy as np

# custom imports
from SDCA.sdca4crf.utils import entropy, kullback_leibler, logsubtractexp, subtractexp_scalar


class SequenceMarginals:
    """Represent anything that is decomposable over the nodes and edges of a sequential model.

    It can be a score, a conditional probability p(y|x) under the form of MARGINALS or
    LOG-MARGINALS (in which case self.islog=True), the ascent direction, the derivative of the KL
    or the entropy."""

    def __init__(self, unary, binary, log):
        self.length = unary.shape[0]
        self.nb_labels = unary.shape[1]
        self.unary = unary
        self.binary = binary
        self.islog = log

        if self.length == 0:
            raise ValueError("Sequences of length 0 are not accepted.")

        if self.length != binary.shape[0] + 1:
            raise ValueError("Wrong length of marginals: %i vs %i"
                             % (unary.shape[0], binary.shape[0] + 1))

        if self.nb_labels != binary.shape[1] \
                or self.nb_labels != binary.shape[2]:
            raise ValueError("Wrong alphabet size: %i vs (%i, %i)"
                             % (unary.shape[1], binary.shape[1], binary.shape[2]))

    def __str__(self):
        return "unary: \n" + np.array_str(self.unary) \
               + "\n binary: \n" + np.array_str(self.binary)

    def __repr__(self):
        return "unary: \n" + np.array_repr(self.unary) \
               + "\n binary: \n" + np.array_repr(self.binary)

    def display(self, alphabet):
        alength = len(alphabet)
        plt.matshow(self.unary)
        plt.xticks(range(alength), [alphabet[x] for x in range(alength)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("unary marginals")
        if self.length > 1:
            plt.matshow(self.binary.sum(axis=0))
            plt.xticks(range(alength), [alphabet[x] for x in range(alength)])
            plt.yticks(range(alength), [alphabet[x] for x in range(alength)])
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title("sum of binary marginals")

    #########################################
    # Special operations
    #########################################
    def log(self):
        return SequenceMarginals(np.log(self.unary), np.log(self.binary), log=True)

    def exp(self):
        return SequenceMarginals(np.exp(self.unary), np.exp(self.binary), log=False)

    def log_reduce_exp(self, to_add):
        if self.length == 1:  # the joint is the unary
            themax = np.amax(self.unary)
            return themax + np.log(np.sum(np.exp(self.unary - themax)
                                          + to_add * np.exp(-themax)))
        elif self.length == 2:  # the joint is the binary
            themax = np.amax(self.binary)
            return themax + np.log(np.sum(np.exp(self.binary - themax))
                                   + to_add * np.exp(-themax))
        else:
            themax = max(np.amax(self.unary[1:-1]), np.amax(self.binary))
            return themax + np.log(np.sum(np.exp(self.binary - themax))
                                   - np.sum(np.exp(self.unary[1:-1] - themax))
                                   + to_add * np.exp(-themax))

    def convex_combination(self, other, s):
        """Return (1-s)*self + s*other"""
        if s == 0:
            return self
        if s == 1:
            return other

        if self.islog:
            unary = np.logaddexp(np.log(1 - s) + self.unary, np.log(s) + other.unary)
            binary = np.logaddexp(np.log(1 - s) + self.binary, np.log(s) + other.binary)
        else:
            unary = (1 - s) * self.unary + s * other.unary
            binary = (1 - s) * self.binary + s * other.binary

        return SequenceMarginals(unary=unary, binary=binary, log=self.islog)

    def logsubtractexp(self, other):
        """Return the ascent direction without numerical issue"""
        unary, usign = logsubtractexp(self.unary, other.unary)
        binary, bsign = logsubtractexp(self.binary, other.binary)

        logvalue = SequenceMarginals(unary=unary, binary=binary, log=True)
        signs = SequenceMarginals(unary=usign, binary=bsign, log=False)
        return logvalue, signs

    #########################################
    # Typical arithmetic operations
    #########################################
    def combine(self, other, ufunc):
        unary = ufunc(self.unary, other.unary)
        binary = ufunc(self.binary, other.binary)
        return SequenceMarginals(unary, binary, self.islog)

    def subtract(self, other):
        return self.combine(other, np.subtract)

    def multiply(self, other):
        return self.combine(other, np.multiply)

    def multiply_scalar(self, scalar):
        return SequenceMarginals(scalar * self.unary, scalar * self.binary, self.islog)

    #########################################
    # Assertion operations
    #########################################
    def is_density(self, integral=1):

        return np.isclose(np.sum(self.unary, axis=1), integral).all() \
               and np.isclose(np.sum(self.binary, axis=(1, 2)), integral).all()

    def is_consistent(self):
        if self.length == 1:
            return True
        ans = True
        from_left_binary = np.sum(self.binary, axis=1)
        from_right_binary = np.sum(self.binary, axis=2)
        if not np.isclose(from_left_binary, self.unary[1:]).all():
            ans = False
            print("Left inconsistent with unary.")
        if not np.isclose(from_right_binary, self.unary[:-1]).all():
            ans = False
            print("Right inconsistent with unary.")
        if not np.isclose(from_right_binary[1:], from_left_binary[:-1]).all():
            ans = False
            print("Left inconsistent with right.")
        return ans

    #########################################
    # Information theory
    #########################################
    def entropy(self):
        returnlog = False
        if self.length == 1:
            return entropy(self.unary, returnlog=returnlog)

        elif self.length == 2:
            return entropy(self.binary, returnlog=returnlog)

        else:
            cliques = entropy(self.binary, returnlog=True)
            separations = entropy(self.unary[1:-1], returnlog=True)
            return subtractexp_scalar(cliques, separations)

    def kullback_leibler(self, other):
        returnlog = False
        if self.length != other.length:
            raise ValueError("Not the same sequence length %i %i" % (self.length, other.length))

        if self.length == 1:
            return kullback_leibler(self.unary, other.unary, returnlog=returnlog)

        elif self.length == 2:
            return kullback_leibler(self.binary, other.binary, returnlog=returnlog)

        else:
            cliques = kullback_leibler(self.binary, other.binary, returnlog=True)
            separations = kullback_leibler(self.unary[1:-1], other.unary[1:-1],
                                           returnlog=True)
            return subtractexp_scalar(cliques, separations)
