import warnings
from time import time

import numpy as np
from pystruct.learners.ssvm import BaseSSVM
from pystruct.utils import find_constraint
from sklearn.utils import check_random_state


class FrankWolfeSSVMLocal(BaseSSVM):

    def __init__(self, model, max_iter=1000, c=1.0, verbose=0, n_jobs=1,
                 show_loss_every=0, logger=None, batch_mode=False,
                 line_search=True, check_dual_every=10, tol=.001,
                 do_averaging=True, sample_method='perm', random_state=None):

        if n_jobs != 1:
            warnings.warn("FrankWolfeSSVM does not support multiprocessing"
                          " yet. Ignoring n_jobs != 1.")

        if sample_method not in ['perm', 'rnd', 'seq']:
            raise ValueError("sample_method can only be perm, rnd, or seq")

        BaseSSVM.__init__(self, model, max_iter, c, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          logger=logger)
        self.tol = tol
        self.batch_mode = batch_mode
        self.line_search = line_search
        self.check_dual_every = check_dual_every
        self.do_averaging = do_averaging
        self.sample_method = sample_method
        self.random_state = random_state

    def _calc_dual_gap(self, param_x, param_y):
        n_samples = len(param_x)
        joint_feature_gt = self.model.batch_joint_feature(param_x, param_y, param_y)
        y_hat = self.model.batch_loss_augmented_inference(param_x, param_y, self.w,
                                                          relaxed=True)
        djoint_feature = joint_feature_gt - \
            self.model.batch_joint_feature(param_x, y_hat)
        ls = np.sum(self.model.batch_loss(param_y, y_hat))
        ws = djoint_feature * self.C
        l_rescaled = self.param_l * n_samples * self.C
        dual_val = -0.5 * np.sum(self.w ** 2) + l_rescaled
        w_diff = self.w - ws
        dual_gap = w_diff.T.dot(self.w) - l_rescaled + ls * self.C
        primal_val = dual_val + dual_gap
        return dual_val, dual_gap, primal_val

    def _frank_wolfe_batch(self, param_x, param_y):
        param_l = 0.0
        n_samples = float(len(param_x))
        joint_feature_gt = self.model.batch_joint_feature(param_x, param_y, param_y)

        for iteration in range(self.max_iter):
            y_hat = self.model.batch_loss_augmented_inference(param_x, param_y, self.w,
                                                              relaxed=True)
            djoint_feature = joint_feature_gt - \
                self.model.batch_joint_feature(param_x, y_hat)
            ls = np.mean(self.model.batch_loss(param_y, y_hat))
            ws = djoint_feature * self.C

            w_diff = self.w - ws
            dual_gap = 1.0 / (self.C * n_samples) * w_diff.T.dot(self.w) - param_l + ls

            if self.line_search:
                eps = 1e-15
                gamma = dual_gap / (np.sum(w_diff ** 2) /
                                    (self.C * n_samples) + eps)
                gamma = max(0.0, min(1.0, gamma))
            else:
                gamma = 2.0 / (iteration + 2.0)

            dual_val = -0.5 * np.sum(self.w ** 2) + param_l * (n_samples * self.C)
            dual_gap_display = dual_gap * n_samples * self.C
            primal_val = dual_val + dual_gap_display

            self.primal_objective_curve_.append(primal_val)
            self.objective_curve_.append(dual_val)
            self.timestamps_.append(time() - self.timestamps_[0])
            if self.verbose > 0:
                print(("iteration %d, dual: %f, dual_gap: %f, primal: %f, gamma: %f"
                       % (iteration, dual_val, dual_gap_display, primal_val, gamma)))

            self.w = (1.0 - gamma) * self.w + gamma * ws
            param_l = (1.0 - gamma) * param_l + gamma * ls

            if self.logger is not None:
                self.logger(self, iteration)

            if dual_gap < self.tol:
                return

    def _frank_wolfe_bc(self, param_x, param_y, initialize=True):

        n_samples = len(param_x)
        w = self.w.copy()
        if initialize:
            self.w_mat = np.zeros((n_samples, self.model.size_joint_feature))
            self.l_mat = np.zeros(n_samples)
            self.l_loss = 0.0
            self.k = 0
            self.rng = check_random_state(self.random_state)

        for iteration in range(self.max_iter):
            if self.verbose > 0:
                print(("Iteration %d" % iteration))

            perm = np.arange(n_samples)
            if self.sample_method == 'perm':
                self.rng.shuffle(perm)
            elif self.sample_method == 'rnd':
                perm = self.rng.randint(low=0, high=n_samples, size=n_samples)

            for j in range(n_samples):
                i = perm[j]
                x, y = param_x[i], param_y[i]
                y_hat, delta_joint_feature, slack, loss = find_constraint(
                    self.model, x, y, w)
                ws = delta_joint_feature * self.C
                ls = loss / n_samples
                if self.line_search:
                    eps = 1e-15
                    w_diff = self.w_mat[i] - ws
                    self.gamma = (w_diff.T.dot(w)
                                  - (self.C * n_samples) * (self.l_mat[i] - ls)) / (np.sum(w_diff ** 2) + eps)
                    self.gamma = max(0.0, min(1.0, self.gamma))
                else:
                    self.gamma = 2.0 * n_samples / (self.k + 2.0 * n_samples)

                w -= self.w_mat[i]
                self.w_mat[i] = (1.0 - self.gamma) * \
                    self.w_mat[i] + self.gamma * ws
                w += self.w_mat[i]

                self.l_loss -= self.l_mat[i]
                self.l_mat[i] = (1.0 - self.gamma) * \
                    self.l_mat[i] + self.gamma * ls
                self.l_loss += self.l_mat[i]

                if self.do_averaging:
                    self.rho = 2. / (self.k + 2.)
                    self.w = (1. - self.rho) * self.w + self.rho * w
                    self.param_l = (1. - self.rho) * self.param_l + self.rho * self.l_loss
                else:
                    self.w = w
                    self.param_l = self.l_loss
                self.k += 1

            if (self.check_dual_every != 0) and (iteration % self.check_dual_every == 0):
                dual_val, dual_gap, primal_val = self._calc_dual_gap(param_x, param_y)
                self.primal_objective_curve_.append(primal_val)
                self.objective_curve_.append(dual_val)
                self.timestamps_.append(time() - self.timestamps_[0])
                if self.verbose > 0:
                    print(("dual: %f, dual_gap: %f, primal: %f"
                           % (dual_val, dual_gap, primal_val)))

            if self.logger is not None:
                self.logger(self, iteration)

            if dual_gap < self.tol:
                return

    def fit(self, param_x, param_y, constraints=None, initialize=True):
        if initialize:
            print("initialize == True")
            self.model.initialize(param_x, param_y)
            self.objective_curve_, self.primal_objective_curve_ = [], []
            self.timestamps_ = [time()]
            self.w = getattr(self, "w", np.zeros(
                self.model.size_joint_feature))
            self.param_l = getattr(self, "param_l", 0)
        try:
            if self.batch_mode:
                self._frank_wolfe_batch(param_x, param_y)
            else:
                self._frank_wolfe_bc(param_x, param_y, initialize)
        except KeyboardInterrupt:
            pass
        if self.verbose:
            print("Calculating final objective.")
        self.timestamps_.append(time() - self.timestamps_[0])
        self.primal_objective_curve_.append(self._objective(param_x, param_y))
        self.objective_curve_.append(self.objective_curve_[-1])
        if self.logger is not None:
            self.logger(self, 'final')
        return self
