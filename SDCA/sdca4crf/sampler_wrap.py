import numpy as np

from SDCA.sdca4crf.parameters.radius import radii
from SDCA.sdca4crf.sampler import Sampler


class SamplerWrap:
    UNIFORM = 0
    IMPORTANCE = 1
    GAP = 2
    GAPP = 3
    MAX = 4
    SAFEMAX = 5

    def __init__(self, sampling_scheme, non_uniformity,
                 gaps_array, trainset, regularization):

        self.size = len(trainset)
        self.non_uniformity = non_uniformity

        if sampling_scheme == "uniform":
            self.scheme = SamplerWrap.UNIFORM
        elif sampling_scheme == "importance":
            self.scheme = SamplerWrap.IMPORTANCE
        elif sampling_scheme == "gap":
            self.scheme = SamplerWrap.GAP
        elif sampling_scheme == "gap+":
            self.scheme = SamplerWrap.GAPP
        elif sampling_scheme == "max":
            self.scheme = SamplerWrap.MAX
        elif sampling_scheme == "safemax":
            self.scheme = SamplerWrap.SAFEMAX
            self.importances = np.ones(self.size)
            self.sampler = SafeMaxSampler(gaps_array, trainset, regularization)
        else:
            raise ValueError(" %s is not a valid argument for sampling scheme" % str(
                sampling_scheme))

        if self.scheme in [SamplerWrap.UNIFORM, SamplerWrap.GAP,
                           SamplerWrap.MAX, SamplerWrap.SAFEMAX]:
            self.importances = np.ones(self.size)
        elif self.scheme in [SamplerWrap.IMPORTANCE, SamplerWrap.GAPP]:
            self.importances = 1 + \
                radii(trainset) ** 2 / self.size / regularization

        if self.scheme == SamplerWrap.SAFEMAX:
            self.sampler = SafeMaxSampler(gaps_array, trainset, regularization)
        else:
            self.sampler = Sampler(gaps_array * self.importances,
                                   is_determinist=(self.scheme == SamplerWrap.MAX))

    def update(self, sample_id, individual_gap, primal_direction_norm, step_size):
        if self.scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler.update(
                individual_gap * self.importances[sample_id], sample_id)
        elif self.scheme == SamplerWrap.SAFEMAX:
            self.sampler.update(sample_id, individual_gap,
                                primal_direction_norm, step_size)

    def full_update(self, gaps_array):
        if self.scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler = Sampler(gaps_array * self.importances)
        elif self.scheme == SamplerWrap.SAFEMAX:
            self.sampler.full_update(gaps_array)

    def sample(self):
        if np.random.rand() > self.non_uniformity:  # then sample uniformly
            return np.random.randint(self.size)
        else:  # sample proportionally to the duality gaps
            return self.sampler.sample()


class SafeMaxSampler:

    def __init__(self, gaps_array, trainset, regularization):
        self.gaps = gaps_array
        self.radii = radii(trainset)
        self.size = len(gaps_array)
        self.regularization = regularization

    def update(self, sample_id, individual_gap, primal_direction_norm, step_size):
        self.gaps -= (2 * step_size * primal_direction_norm * self.radii
                      / self.regularization / self.size)
        self.gaps[sample_id] = individual_gap

    def full_update(self, gaps_array):
        self.gaps = gaps_array

    def sample(self):
        return np.argmax(self.gaps)
