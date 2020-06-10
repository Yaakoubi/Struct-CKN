import pickle
import time

import numpy as np
import tensorboard_logger as tl
import math


def are_consistent(monitor_dual, monitor_all):
    if math.isnan(monitor_dual.get_value()) or math.isnan(monitor_all.dual_objective):
        return True
    return np.isclose(monitor_dual.get_value(), monitor_all.dual_objective)


class MonitorAllObjectives:

    def __init__(self, regularization, weights, marginals, ground_truth_centroid, trainset,
                 testset, use_tensorboard):

        self.regularization = regularization
        self.ntrain = len(trainset)  # size of training set
        self.trainset = trainset
        self.ground_truth_centroid = ground_truth_centroid

        # compute the primal and dual objectives and compare them with the duality gap
        self.primal_objective = None
        self.dual_objective = None
        self.array_gaps = None
        self.duality_gap = None

        # compute the test error if a test set is present:
        self.testset = testset
        self.loss01 = None
        self.hamming = None
        if self.testset is not None:
            ntest = len(testset)
            self.update_test_error(weights)
        else:
            ntest = 0

        # logging
        self.use_tensorboard = use_tensorboard

        # save values in a dictionary
        self.results = {
            "number of training samples": self.ntrain,
            "number of test samples": ntest,
            "steps": [],
            "times": [],
            "primal objectives": [],
            "dual objectives": [],
            "duality gaps": [],
            "0/1 loss": [],
            "hamming loss": []
        }

        # time spent monitoring the objectives
        self.delta_time = time.time()

        # compute all the values once - EXPENSIVE
        self.full_batch_update(weights, marginals, step=0, count_time=False)

    def __repr__(self):
        return """regularization: {} \t nb train: {} \n
               Primal - Dual = {} - {} = {} \t Duality gap =  {} \n
               Test error 0/1 = {} \t Hamming = {}""".format(
            self.regularization,
            self.ntrain,
            self.primal_objective,
            self.dual_objective,
            self.primal_objective - self.dual_objective,
            self.duality_gap,
            self.loss01,
            self.hamming
        )

    def full_batch_update(self, weights, marginals, step, count_time):
        t1 = time.time()
        gaps_array = self.update_objectives(weights, marginals)
        t2 = time.time()
        if not count_time:  # in case of sampler update
            self.delta_time += t2 - t1

        self.update_test_error(weights)
        self.delta_time += time.time() - t2

        self.append_results(step)
        self.log_tensorboard(step)

        return gaps_array

    def update_objectives(self, weights, marginals):
        weights_squared_norm = weights.squared_norm()

        entropy = sum(margs.entropy() for margs in marginals) / self.ntrain
        self.dual_objective = entropy - self.regularization / 2 * weights_squared_norm

        gaps_array = np.empty(self.ntrain, dtype=float)
        sum_log_partitions = 0
        for i in range(self.ntrain):
            newmargs, log_partition = weights.infer_probabilities(
                self.trainset.get_points_sequence(i))
            gaps_array[i] = marginals[i].kullback_leibler(newmargs)
            sum_log_partitions += log_partition

        # calculate the primal score
        # take care that the log-partitions are not based on the corrected features (ground
        # truth minus feature) but on the raw features.
        self.primal_objective = \
            self.regularization / 2 * weights_squared_norm \
            + sum_log_partitions / self.ntrain \
            - weights.inner_product(self.ground_truth_centroid)

        # update the value of the duality gap
        self.duality_gap = gaps_array.mean()

        assert self.check_duality_gap(), self

        return gaps_array

    def check_duality_gap(self):
        if math.isnan(self.duality_gap) or math.isnan(self.dual_objective):
            return True
        """Check that the objective values are coherent with each other."""
        return np.isclose(self.duality_gap, self.primal_objective - self.dual_objective)

    def update_test_error(self, weights):
        if self.testset is None:
            return

        self.loss01 = 0
        self.hamming = 0
        total_labels = 0

        for point, label in self.testset:
            prediction = weights.predict(point)
            tmp = np.sum(label != prediction)
            self.hamming += tmp
            self.loss01 += (tmp > 0)
            total_labels += len(label)

        self.loss01 /= len(self.testset)
        self.hamming /= total_labels

    def append_results(self, step):
        self.results["steps"].append(step)
        self.results["times"].append(time.time() - self.delta_time)
        self.results["primal objectives"].append(self.primal_objective)
        self.results["dual objectives"].append(self.dual_objective)
        self.results["duality gaps"].append(self.duality_gap)
        if self.testset is not None:
            self.results["0/1 loss"].append(self.loss01)
            self.results["hamming loss"].append(self.hamming)

    def save_results(self, logdir):
        with open(logdir + "/objectives.pkl", "wb") as out:
            pickle.dump(self.results, out)

    def log_tensorboard(self, step):
        if self.use_tensorboard:
            tl.log_value("log10 duality gap", np.log10(self.duality_gap), step)
            tl.log_value("primal objective", self.primal_objective, step)
            tl.log_value("dual objective", self.dual_objective, step)
            if self.testset is not None:
                tl.log_value("01 loss", self.loss01, step)
                tl.log_value("hamming loss", self.hamming, step)


class MonitorDualObjective:

    def __init__(self, regularization, weights, marginals):
        self.train_size = len(marginals)
        self.regularization = regularization
        self.entropies = np.array([margs.entropy() for margs in marginals])
        self.entropy = self.entropies.mean()
        self.weights_squared_norm = weights.squared_norm()

        self.dual_objective = self.entropy - \
            self.regularization / 2 * self.weights_squared_norm

    def update(self, i, newmarg_entropy, norm_update):
        self.weights_squared_norm += norm_update

        self.entropy += (newmarg_entropy - self.entropies[i]) / self.train_size
        self.entropies[i] = newmarg_entropy

        self.dual_objective = self.entropy - \
            self.regularization / 2 * self.weights_squared_norm

    def get_value(self):
        return self.dual_objective

    def log_tensorboard(self, step):
        tl.log_value("weights_squared_norm", self.weights_squared_norm, step)
        tl.log_value("entropy", self.entropy, step)
        tl.log_value("dual objective", self.dual_objective, step)


class MonitorDualityGapEstimate:

    def __init__(self, gaps_array):
        self.ntrain = len(gaps_array)
        self.gaps_array = gaps_array
        self.gap_estimate = gaps_array.mean()

    def update(self, i, new_gap):
        self.gap_estimate += (new_gap - self.gaps_array[i]) / self.ntrain
        self.gaps_array[i] = new_gap

    def get_value(self):
        return self.gap_estimate

    def log_tensorboard(self, true_duality_gap, step):
        tl.log_value("log10 duality gap estimate",
                     np.log10(self.gap_estimate), step)
        tl.log_value("gap estimate/ true gap",
                     self.gap_estimate / true_duality_gap, step)


class MonitorSparsity:

    def __init__(self):
        pass

    def log_tensorboard(self, weights, step):
        sparsity = np.mean(weights.emission < 1e-10)
        tl.log_value('sparsity_coefficient', sparsity, step)
        tl.log_histogram('weight matrix', weights.emission.tolist(), step)


class MonitorSpeed:

    def __init__(self, step=0):
        self.previous_step = step
        self.previous_time = time.time()
        self.speed = 0

    def update(self, step):
        current_time = time.time()
        self.speed = (step - self.previous_step) / \
            (current_time - self.previous_time)
        self.previous_step = step
        self.previous_time = current_time

    def log_tensorboard(self):
        tl.log_value("iteration per second", self.speed, self.previous_step)

    def log_time_spent_on_line_search(self, p, step):
        tl.log_value("percent time spend line search", p, step)
