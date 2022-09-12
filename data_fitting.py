# import numpy as xp
import copy
import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    cupy_available = False
else:
    cupy_available = True


def get_backend(ndarray):
    if cupy_available:
        xp = cp.get_array_module(ndarray)
    else:
        xp = np

    return xp


class Einsum(object):
    """Einsum helper functions."""

    @staticmethod
    def transpose(ndarray):
        transpose_definition = '...lm->...ml'

        xp = get_backend(ndarray)
        return xp.einsum(transpose_definition, ndarray)

    @staticmethod
    def matmul(ndarray1, ndarray2):
        matmul_definition = '...ij,...jk->...ik'

        xp = get_backend(ndarray1)
        return xp.einsum(matmul_definition, ndarray1, ndarray2)

    @staticmethod
    def chained_matmul(ndarray1, ndarray2, ndarray3):
        chained_definition = '...ij,...jk,...km->...im'

        xp = get_backend(ndarray1)
        return xp.einsum(chained_definition, ndarray1, ndarray2, ndarray3)

    @staticmethod
    def reduce_dimensions(ndarray):
        reduce_definition = '...lm->...l'

        xp = get_backend(ndarray)
        return xp.einsum(reduce_definition, ndarray)


class LMFit(object):
    """Numpy accelerated LMFit implementation. Vectorized implementation is based off
    Henri P. Gavin's description of the LM fit method.
    http://people.duke.edu/~hpgavin/ce281/lm.pdf

    This implementation extends the vectorization to performing fits on multiple data sets and
    multiple models simultaneously"""

    epsilon_1 = 1e-3    # Convergence criteria - gradient approaching zero
    epsilon_2 = 1e-3    # Convergence criteria - parameters have stagnated
    epsilon_3 = 1e-1    # Convergence criteria - reduced chi-squared value approaching zero
    epsilon_4 = 1e-1    # Acceptance threshold for an LM update step
    epsilon_5 = 2.5e-3  # New constant introduced - threshold for stopping fitting based on newly finished/failed fits

    L_0 = 1e-2          # Initial value for lambda
    L_increase = 11     # Factor to increase lambda by after rejected step
    L_decrease = 9      # Factor to reduce lambda by after accepted step
    L_max = 1e7         # Max value for lambda
    L_min = 1e-7        # Min value for lambda

    delta_p = 0.001     # For computing numerical derivatives

    iter_multiplier = 30    # This times n_params = max number of iterations to do

    def __init__(self, fn, x_data, y_data, params, weights, bounds=None, num_points=None, **kwargs):
        """
        Uses the Levenberg-Marquardt weighted least-squares fitting algorithm to find the optimal parameters which
        minimize the chi-squared value of a non-linear function. This implementation is able to find the optimal
        parameters for many same-length datasets in parallel.

        :param fn:         User-specified model function
        :type  fn:         Callable, with signature fn(x_data, params, **kwargs): model_data[, Jacobian]
        :param x_data:     Independent variable values
        :type  x_data:     ndarray [..., m_points]
        :param y_data:     Dependent variable values
        :type  y_data:     ndarray [..., m_points]
        :param params:     Parameters to vary for fitting
        :type  params:     ndarray [..., n_params]
        :param weights:    Weighting factors
        :type  weights:    ndarray [..., m_points]
        :param bounds:     Upper and lower bounds for params
        :type  bounds:     ndarray [..., n_params, 2]
        :param num_points: Number of data points for each data set.
        :type  num_points: ndarray [...]
        :param kwargs:     Extra arguments as needed for fn
        :type  kwargs:     dict

        The array dimensions of inputs for this function must be broadcastable to one another, as described in
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        """
        super(LMFit, self).__init__()
        xp = get_backend(weights)

        self.fn = fn
        self.kwargs = kwargs

        # The reshaping below is to allow for proper matrix calculations
        self.x_data = x_data[..., xp.newaxis]
        self.y_data = y_data[..., xp.newaxis]
        self.params = params[..., xp.newaxis]
        self.weights = weights[..., xp.newaxis]
        self.bounds = bounds
        self.chi_2 = None

        lm_step_shape = self.params.shape[:-2] + (1, 1)

        # [..., 1, 1]
        self.lm_step = xp.ones(lm_step_shape) * LMFit.L_0

        # Array of flags which indicate if a given dataset has converged
        self.converged = xp.full(lm_step_shape, False)

        # Using matrix operations, it's not possible to selectively perform fits. Masks are used
        # to avoid fitting issues when fits have finished or failed.
        self.fit_mask = xp.full(lm_step_shape, False)

        self.n_params = self.params.shape[-2]

        if num_points is not None:
            tmp_points = xp.array(xp.broadcast_to(num_points[..., xp.newaxis], lm_step_shape))

            # [..., 1, 1]
            self.fit_mask[tmp_points <= 0] = True

            # All datasets with invalid number of points cannot be fitted, so output params are set to input params.
            tmp_points[tmp_points <= 0] = self.n_params

            # Number of data points in each dataset
            self.m_points = tmp_points
        else:
            # Same number of points (m_points) in each dataset
            self.m_points = y_data.shape[-2]

        # Max number of iterations set by number of variable parameters
        n_iters = self.n_params * LMFit.iter_multiplier

        self.cov_mat = None

        i = 0
        prev_num_stopped_fits = 0
        while True:

            if i == n_iters:
                break

            self.levenberg_marquardt_iteration()

            if i == 0:
                # This will be any datasets with num_points <= 0 or that stop on first iteration
                prev_num_stopped_fits = xp.count_nonzero(self.fit_mask)
            else:
                # Number of datasets that converged/failed by the end of this iteration
                num_stopped_fits = xp.count_nonzero(self.fit_mask)
                if num_stopped_fits != 0:
                    # TODO: Does this mean that if any fits previously stopped (such as if num_points <= 0
                    #  off the gun) and no new fits stopped this iteration, it will break?
                    #  e.g. we have prev_num_stopped_fits = x since num_points = 0 for some datasets,
                    #  then when i=1 if no new fits have stopped then change = 1-(x/x) = 0 < epsilon_5,
                    #  so the whole thing stops.
                    change = 1 - (float(prev_num_stopped_fits) / float(num_stopped_fits))
                    # Break if the number of fits which stopped this iteration is small compared to
                    # total number of stopped fits
                    if change < LMFit.epsilon_5:
                        break
                prev_num_stopped_fits = num_stopped_fits

            i += 1

        self.cov_mat = self.compute_cov_mat()
        self.chi_2 = self.chi_2[..., 0, 0]              # Dimensions are [..., 1, 1]
        self.fitted_params = self.params[..., 0]        # Dimensions are [..., n_params, 1]

    def compute_chi2(self, y_yp):
        """
        Calculates the chi_2 values using eqn #2 from the reference document.

        :param      y_yp:  data - model
        :type       y_yp:  ndarray [..., m_points, 1]

        :returns:   The chi_2 results.
        :rtype:     ndarray [..., 1, 1]
        """
        y_ypt = Einsum.transpose(y_yp)
        chi_2 = Einsum.matmul(y_ypt, self.weights * y_yp)

        return chi_2

    def get_model_and_J(self, params):
        """
        Evaluates the model with new parameters. If there is no user supplied function for a
        Jacobian, then the central finite differences method in 4.1.2 is used. Currently each
        iteration will use finite difference, but future improvements will implement the
        optimizations outlined.

        :param      params:  The array of parameters.
        :type       params:  ndarray [..., n_params, 1]

        :returns:   The model and Jacobian.
        :rtype:     dict
        """

        xp = get_backend(self.weights)

        m = self.fn(self.x_data, params, **self.kwargs)

        if isinstance(m, tuple):
            model, J = m
        else:
            # [..., m_points]
            model = m

            J_shape = model.shape[:-1] + (self.n_params,)
            # [..., n_params]
            J = xp.empty(J_shape, dtype=model.dtype)

            dp = self.delta_p * (1 + xp.abs(params))
            for i in range(self.n_params):
                # [..., n_params, m_points]
                dp_i = xp.zeros(params.shape)
                dp_i[..., [i], :] = dp[..., [i], :]

                p_upper = params + dp_i
                p_lower = params - dp_i

                m_upper = self.fn(self.x_data, p_upper, **self.kwargs)
                m_lower = self.fn(self.x_data, p_lower, **self.kwargs)

                # Central difference method
                J[..., [i]] = (m_upper - m_lower) / (2 * dp[..., [i], :])

        return {'model': model, 'J': J}

    def compute_cov_mat(self):
        """
        Calculates the covariance matrix.

        """

        xp = get_backend(self.weights)

        model = self.get_model_and_J(self.params)

        Jt = Einsum.transpose(model['J'])
        Jt_w_J = Einsum.matmul(Jt, self.weights * model['J'])

        diag = xp.arange(Jt_w_J.shape[-1])

        # Set covariance matrix diagonals to 1.0 for all converged datasets
        stopped_grad = xp.full(Jt_w_J.shape, False)
        stopped_grad[..., [diag], [diag]] = ~self.converged
        Jt_w_J[stopped_grad] = 1.0

        return xp.linalg.inv(Jt_w_J)

    def levenberg_marquardt_iteration(self):
        """
        Performs a single step of the LM algorithm.

        """

        xp = get_backend(self.weights)

        model = self.get_model_and_J(self.params)

        y_yp = (self.y_data - model['model'])
        chi_2 = self.compute_chi2(y_yp)

        # Eqn #13
        Jt = Einsum.transpose(model['J'])
        Jt_w_J = Einsum.matmul(Jt, self.weights * model['J'])

        diag = xp.arange(Jt_w_J.shape[-1])

        lm_Jt_w_J_diag = xp.zeros(Jt_w_J.shape)

        lm_Jt_w_J_diag[..., [diag], [diag]] = (self.lm_step * Jt_w_J[..., [diag], [diag]])

        # [..., n_params, n_params]
        grad = Jt_w_J + lm_Jt_w_J_diag

        # When fits converge/diverge such that the gradients tend to 0, this stops the matrix
        # from becoming singular.
        invertible = xp.linalg.cond(grad) < 1 / xp.finfo(grad.dtype).eps
        self.fit_mask[~invertible] = True
        stopped_grad = xp.full(grad.shape, False)
        stopped_grad[..., [diag], [diag]] = self.fit_mask
        grad[stopped_grad] = 1.0

        # [..., n_params, n_params]
        Jt_w_Jinv = xp.linalg.inv(grad)

        # [..., n_params, m_points] x [..., m_points, 1]
        # = [..., n_params, 1]
        Jt_w_yyp = Einsum.matmul(Jt, self.weights * y_yp)

        # [..., n_params, n_params] x [..., n_params, 1]
        # = [..., n_params, 1]
        h_lm = Einsum.matmul(Jt_w_Jinv, Jt_w_yyp)
        del model, y_yp, Jt, Jt_w_J, diag, stopped_grad, grad, Jt_w_Jinv

        # Update the parameters and check whether they are in bounds. This algorithm is primarily
        # meant to fit oscillating functions so we stop fitting once a parameter goes out of bounds.
        # It is very likely that the algorithm will not be able to properly step in the chi_2 space
        # and will never converge.

        # [..., n_params, 1]
        tmp_params = self.params + h_lm

        if self.bounds is not None:
            minimum = self.bounds[..., [0]]
            maximum = self.bounds[..., [1]]

            d_min = tmp_params < minimum
            d_max = tmp_params > maximum

            tmp_params[d_min] = self.params[d_min]
            tmp_params[d_max] = self.params[d_max]

            self.fit_mask[xp.any(d_min, axis=-2)] = True
            self.fit_mask[xp.any(d_max, axis=-2)] = True

        # Eqn #16, calculating rho
        model_new = self.get_model_and_J(tmp_params)

        y_yp_new = self.y_data - model_new['model']

        chi_2_new = self.compute_chi2(y_yp_new)
        self.chi_2 = chi_2_new

        rho_numerator = chi_2 - chi_2_new
        # [..., 1, n_params] x ( [..., n_params, n_params] x [..., n_params, 1] + [..., n_params, 1] )
        # = [..., 1, 1]
        rho_denominator = Einsum.matmul(Einsum.transpose(h_lm), (Einsum.matmul(lm_Jt_w_J_diag, h_lm) + Jt_w_yyp))

        rho = rho_numerator / rho_denominator

        # When fitting stops, this will stop div by zero later.
        self.fit_mask[rho == 0.0] = True
        rho[self.fit_mask] = 1.0

        del lm_Jt_w_J_diag, Jt_w_yyp, chi_2, rho_numerator, rho_denominator

        # using 4.1.1 method #1 for the update
        a = xp.maximum(self.lm_step / LMFit.L_decrease, LMFit.L_min)
        a[rho <= LMFit.epsilon_4] = 0.0

        b = xp.minimum(self.lm_step * LMFit.L_increase, LMFit.L_max)
        b[rho > LMFit.epsilon_4] = 0.0

        self.lm_step = a + b
        rho_mask = (rho > LMFit.epsilon_4) & ~self.fit_mask
        rho_mask = xp.repeat(rho_mask, self.params.shape[-2], axis=-2)

        # Accept new parameters for iterations which bettered the rho metric significantly
        self.params[rho_mask] = tmp_params[rho_mask]

        del a, b, rho_mask

        # 4.1.3 convergence criteria
        Jt = Einsum.transpose(model_new['J'])
        Jt_w_yyp = Einsum.matmul(Jt, self.weights * y_yp_new)

        convergence_1 = xp.abs(Jt_w_yyp).max(axis=-2, keepdims=True) < LMFit.epsilon_1

        convergence_2 = xp.abs(h_lm / rho).max(axis=-2, keepdims=True) < LMFit.epsilon_2

        convergence_3 = (chi_2_new / (self.m_points - self.n_params + 1)) < LMFit.epsilon_3

        self.converged |= convergence_1 | convergence_2 | convergence_3

        self.fit_mask[self.converged] = True
