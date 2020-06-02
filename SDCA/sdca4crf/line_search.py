import numpy as np
import tensorboard_logger as tl

from scipy.optimize import minimize_scalar


class LineSearch:

    def __init__(self, weights_dot_primaldir, primaldir_squared_norm, log_dual_direction,
                 alpha_i, beta_i, divergence_gap, args, previous_step_size):

        # linse search direction
        self.alpha_i = alpha_i
        self.beta_i = beta_i
        self.log_dual_direction_squared = log_dual_direction.multiply_scalar(2)

        # values for the derivative of the entropy
        self.divergence_gap = divergence_gap
        self.reverse_gap = self.beta_i.kullback_leibler(alpha_i)

        # values for the quadratic term
        self.primaldir_squared_norm = primaldir_squared_norm
        self.weights_dot_primaldir = weights_dot_primaldir
        self.regularization = args.regularization
        self.train_size = args.train_size

        self.quadratic_coeff = \
            - args.regularization * args.train_size / 2 \
            * self.primaldir_squared_norm

        # hyperparameter
        self.subprecision = args.subprecision
        self.use_scipy = args.use_scipy_optimize
        self.initial_point = previous_step_size if args.init_previous_step_size else 0.5
        self.previous_step_size = previous_step_size
        self.skip = args.skip_line_search

        # return values
        self.optimal_step_size = 0
        self.subobjectives = []

    def evaluator(self, step_size, return_f=False, return_df=False, return_newton=False):
        """Return the line search function f and its derivative df evaluated in the step
        size step_size. Because the second derivative can be insanely large, we return instead the
        Newton step f'(x)/f''(x). Newton can be returned only if df is returned."""
        ans = []

        newmarg = self.new_marginal(step_size)

        if return_f:
            ans.append(self._function(newmarg, step_size))

        if return_df:
            df = self._derivative(newmarg, step_size)
            ans.append(df)

            if df != 0 and return_newton:
                ans.append(self._newton(newmarg, df))

        return tuple(ans) if len(ans) > 1 else ans[0]

    def new_marginal(self, step_size):
        return self.alpha_i.convex_combination(self.beta_i, step_size)

    def _function(self, newmarg, step_size):
        quadratic = - self.train_size * self.regularization / 2 * self.norm_update(step_size)
        return newmarg.entropy() + quadratic

    def norm_update(self, step_size):
        return step_size ** 2 * self.primaldir_squared_norm \
               + 2 * step_size * self.weights_dot_primaldir

    def _derivative(self, newmarg, step_size):
        if step_size == 0:
            return self.divergence_gap + self.reverse_gap
        elif step_size == 1:
            return 2 * self.quadratic_coeff
        else:
            return self.divergence_gap \
                   + self.beta_i.kullback_leibler(newmarg) \
                   - self.alpha_i.kullback_leibler(newmarg) \
                   + 2 * step_size * self.quadratic_coeff

    def _newton(self, newmarg, df):
        log_ddf = self.log_dual_direction_squared \
            .subtract(newmarg) \
            .log_reduce_exp(- 2 * self.quadratic_coeff)  # stable log sum exp
        ans = np.log(np.absolute(df)) - log_ddf  # log(|f'(x)/f''(x)|)
        ans = - np.sign(df) * np.exp(ans)  # f'(x)/f''(x)
        return ans

    def run(self):
        if self.skip:
            if self.evaluator(0, return_f=True) <= \
                    self.evaluator(self.previous_step_size, return_f=True):
                self.optimal_step_size = self.previous_step_size
                self.subobjectives = []  # so that the number of steps is 0
                return self.previous_step_size

        if self.use_scipy:
            return self.run_scipy()
        else:
            return self.run_custom()

    def run_custom(self):
        u0 = self.evaluator(0, return_df=True)
        if u0 <= 0:
            self.optimal_step_size = 0
            self.subobjectives = [u0]
            return 0

        u1 = self.evaluator(1, return_df=True)
        if u1 >= 0:
            self.optimal_step_size = 1
            self.subobjectives = [u1]
            return 1

        self.optimal_step_size, self.subobjectives = safe_newton(
            lambda x: self.evaluator(x, return_df=True, return_newton=True),
            lowerbound=0, upperbound=1, initial_point=self.initial_point,
            u_lower=u0, u_upper=u1, precision=self.subprecision)
        return self.optimal_step_size

    def run_scipy(self):
        result = minimize_scalar(lambda x: - self.evaluator(x, return_f=True),
                                 bounds=(0, 1), method='bounded',
                                 options={'xatol': self.subprecision})

        self.optimal_step_size = result.x
        # hack to simulate a list of the right length
        self.subobjectives = [-result.fun] * result.nfev

        return self.optimal_step_size

    def run_best(self):
        step_scipy = self.run_scipy()
        score_scipy = self.subobjectives[-1]
        step_custom = self.run_custom()
        score_custom = self.evaluator(step_custom, return_f=True)

        if score_custom > score_scipy:
            return step_custom
        else:
            self.optimal_step_size = step_scipy
            self.subobjectives = [score_scipy]
            return step_scipy

    def log_tensorboard(self, step):
        tl.log_value("optimal step size", self.optimal_step_size, step)
        tl.log_value("number of line search steps", len(self.subobjectives), step)
        tl.log_value("log10 primal_direction_squared_norm", np.log10(self.primaldir_squared_norm),
                     step=step)


def safe_newton(evaluator, lowerbound, upperbound, initial_point,
                u_lower, u_upper, precision, max_iter=100):
    """Using a combination of Newton-Raphson and bisection, find the root of a function bracketed
    between lowerbound and upperbound.

    :param scalar function evaluator: user-supplied routine that returns both the function value
    u(x) and u(x)/u'(x)
    :param float lowerbound: point smaller than the root
    :param float upperbound: point larger than the root
    :param float precision: accuracy on the root value rts
    :return: The root, returned as the value rts
    """

    if (u_lower > 0 and u_upper > 0) or (u_lower < 0 and u_upper < 0):
        raise ValueError("Root must be bracketed in [lower bound, upper bound]")
    if u_lower == 0:
        return lowerbound
    if u_upper == 0:
        return upperbound

    if u_lower < 0:  # Orient the search so that f(xl) < 0.
        xl, xh = lowerbound, upperbound
    else:
        xh, xl = lowerbound, upperbound

    rts = initial_point  # Initialize the guess for root
    dxold = abs(upperbound - lowerbound)  # the “stepsize before last"
    dx = dxold  # and the last step

    u, newton = evaluator(rts)
    obj = [u]

    for _ in np.arange(max_iter):  # Loop over allowed iterations.

        rts -= newton  # Newton step
        if (rts - xh) * (rts - xl) <= 0 and abs(newton) <= abs(dxold) / 2:
            # Keep the Newton step  if it remains in the bracket and
            # if it is converging fast enough.
            # This will be false if fdf is NaN.
            dxold = dx
            dx = newton

        else:  # Bisection otherwise
            dxold = dx
            dx = (xh - xl) / 2
            rts = xl + dx

        if abs(dx) < precision:  # Convergence criterion.
            return rts, obj
        u, newton = evaluator(rts)
        # the one new function evaluation per iteration
        obj.append(u)

        if u < 0:  # maintain the bracket on the root
            xl = rts
        else:
            xh = rts

    raise RuntimeError("Maximum number of iterations exceeded in safe_newton")
