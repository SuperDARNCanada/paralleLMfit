#import numpy as xp
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
        return xp.einsum(chained_definition , ndarray1, ndarray2, ndarray3)

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

    epsilon_1 = 1e-3
    epsilon_2 = 1e-3
    epsilon_3 = 1e-1
    epsilon_4 = 1e-1
    epsilon_5 = 2.5e-3 # new constant introduced.

    L_0 = 1e-2
    L_increase = 11
    L_decrease = 9
    L_max = 1e7
    L_min = 1e-7

    delta_p = 0.001

    iter_multiplier = 30


    def __init__(self, fn, x_data, y_data, params, weights, bounds=None, jac=None, num_points=None, **kwargs):
        super(LMFit, self).__init__()
        xp = get_backend(weights)

        self.fn = fn
        self.jac = jac
        self.kwargs = kwargs

        self.x_data = x_data
        self.y_data = y_data
        self.params = params
        self.weights = weights
        self.bounds = bounds



        lm_step_shape = self.params.shape[:-2] + (1,1)
        self.lm_step = xp.ones(lm_step_shape) * LMFit.L_0

        self.converged = xp.full(lm_step_shape, False)

        # Using matrix operations, it's not possible to selectively perform fits. Masks are used
        # to avoid fitting issues when fits have finished or failed.
        self.fit_mask = xp.full(lm_step_shape, False)

        self.n_params = self.params.shape[-2]

        if num_points is not None:
            tmp_points = xp.array(xp.broadcast_to(num_points, lm_step_shape))
            self.fit_mask[tmp_points<=0] = True

            tmp_points[tmp_points<=0] = self.n_params
            self.m_points = tmp_points
        else:
            self.m_points = y_data.shape[-2]

        n_iters = self.n_params * LMFit.iter_multiplier

        self.cov_mat = None

        i = 0
        prev_num_stopped_fits = 0
        while True:

            if i == n_iters:
                break

            print("Running step: ", i)
            self.levenburg_marquardt_iteration()

            if i == 0:
                prev_num_stopped_fits = xp.count_nonzero(self.fit_mask)
            else:
                num_stopped_fits = xp.count_nonzero(self.fit_mask)
                if num_stopped_fits != 0:
                    change = 1 - (float(prev_num_stopped_fits) / float(num_stopped_fits))
                    if change < LMFit.epsilon_5:
                        break
                prev_num_stopped_fits = num_stopped_fits

            i += 1

        self.compute_cov_mat()
        self.fitted_params = self.params


    def compute_chi2(self, y_yp):
        """
        Calculates the chi_2 values using eqn #2 from the reference document.

        :param      y_yp:  data - model
        :type       y_yp:  ndarray [...,n_points,1]

        :returns:   The chi_2 results.
        :rtype:     ndarray [...,1,1]
        """

        xp = get_backend(self.weights)

        y_ypt = Einsum.transpose(y_yp)
        chi_2 = Einsum.chained_matmul(y_ypt, self.weights, y_yp)

        return chi_2

    def get_model_and_J(self, params):
        """
        Evaluates the model with new parameters. If there is no user supplied function for a
        Jacobian, then the central finite differences method in 4.1.2 is used. Currently each
        iteration will use finite difference, but future improvements will implement the
        optimizations outlined.

        :param      params:  The array of parameters.
        :type       params:  ndarray [...,n_params,1]

        :returns:   The model and Jacobian.
        :rtype:     dict
        """

        xp = get_backend(self.weights)

        m = self.fn(self.x_data, params, **self.kwargs)

        if isinstance(m, tuple):
            model, J = m
        else:
            model = m

            J_shape = model.shape[:-1] + (self.n_params,)
            J = xp.empty(J_shape, dtype=model.dtype)

            dp = self.delta_p * (1 + xp.abs(params))
            for i in range(self.n_params):
                dp_i = xp.zeros(params.shape)
                dp_i[...,[i],:] = dp[...,[i],:]

                p_upper = params + dp_i
                p_lower = params - dp_i

                m_upper = self.fn(self.x_data, p_upper, **self.kwargs)
                m_lower = self.fn(self.x_data, p_lower, **self.kwargs)

                J[...,[i]]  = (m_upper - m_lower) / (2 * dp[...,[i],:])

        return {'model' : model, 'J' : J}


    def compute_cov_mat(self):
        """
        Calculates the covariance matrix.

        """

        xp = get_backend(self.weights)

        model = self.get_model_and_J(self.params)

        Jt = Einsum.transpose(model['J'])
        Jt_w = Einsum.matmul(Jt, self.weights)
        Jt_w_J = Einsum.matmul(Jt_w, model['J'])

        diag = xp.arange(Jt_w_J.shape[-1])

        stopped_grad = xp.full(Jt_w_J.shape, False)
        stopped_grad[...,[diag],[diag]] = ~self.converged
        Jt_w_J[stopped_grad] = 1.0

        self.cov_mat = xp.linalg.inv(Jt_w_J)

    def levenburg_marquardt_iteration(self):
        """
        Performs a single step of the LM algorithm.

        """

        xp = get_backend(self.weights)

        model = self.get_model_and_J(self.params)

        y_yp = (self.y_data - model['model'])
        chi_2 = self.compute_chi2(y_yp)

        # Eqn #13
        Jt = Einsum.transpose(model['J'])
        Jt_w = Einsum.matmul(Jt, self.weights)
        Jt_w_J = Einsum.matmul(Jt_w, model['J'])

        diag = xp.arange(Jt_w_J.shape[-1])

        lm_Jt_w_J_diag = xp.zeros(Jt_w_J.shape)

        lm_Jt_w_J_diag[...,[diag],[diag]] = (self.lm_step * Jt_w_J[...,[diag],[diag]])

        grad = Jt_w_J + lm_Jt_w_J_diag


        # When fits converge/diverge such that the gradients tend to 0, this stops the matrix
        # from becoming singular.
        invertable = xp.linalg.cond(grad) < 1/xp.finfo(grad.dtype).eps
        self.fit_mask[~invertable] = True
        stopped_grad = xp.full(grad.shape, False)
        stopped_grad[...,[diag],[diag]] = self.fit_mask
        grad[stopped_grad] = 1.0


        Jt_w_Jinv = xp.linalg.inv(grad)
        Jt_w_yyp = Einsum.matmul(Jt_w, y_yp)
        h_lm = Einsum.matmul(Jt_w_Jinv, Jt_w_yyp)
        del model, y_yp, Jt, Jt_w, Jt_w_J, diag, stopped_grad, grad, Jt_w_Jinv

        # Update the parameters and check whether they are in bounds. This algorithm is primarily
        # meant to fit oscillating functions so we stop fitting once a parameter goes out of bounds.
        # It is very likely that the algorithm will not be able to properly step in the chi_2 space
        # and will never converge.
        tmp_params = self.params + h_lm

        if self.bounds:
            minimum = bounds[0]
            maximum = bounds[1]

            d_min = tmp_params < minimum
            d_max = tmp_params > maximum

            tmp_params[d_min] = self.params[d_min]
            tmp_params[d_max] = self.params[d_max]

            self.fit_mask[xp.any(d_min)] = True
            self.fit_mask[xp.any(d_max)] = True




        # Eqn #16, calculating rho
        model_new = self.get_model_and_J(tmp_params)

        y_yp_new = self.y_data - model_new['model']

        chi_2_new = self.compute_chi2(y_yp_new)
        self.chi_2 = chi_2_new

        rho_numerator = chi_2 - chi_2_new
        rho_denominator = Einsum.matmul(Einsum.transpose(h_lm),
                                        (Einsum.matmul(lm_Jt_w_J_diag, h_lm) + Jt_w_yyp))

        rho = rho_numerator / rho_denominator

        # When fitting stops, this will stop div by zero later.
        self.fit_mask[rho == 0.0] = True
        rho[self.fit_mask] = 1.0

        del lm_Jt_w_J_diag, Jt_w_yyp, chi_2, rho_numerator, rho_denominator



        # using 4.1.1 method #1 for the update
        a = xp.maximum(self.lm_step / LMFit.L_decrease, LMFit.L_min)
        a[rho<=LMFit.epsilon_4] = 0.0

        b = xp.minimum(self.lm_step * LMFit.L_increase, LMFit.L_max)
        b[rho>LMFit.epsilon_4] = 0.0

        self.lm_step = a + b
        rho_mask = (rho > LMFit.epsilon_4) & ~self.fit_mask
        rho_mask = xp.repeat(rho_mask, self.params.shape[-2], axis=-2)

        self.params[rho_mask] = tmp_params[rho_mask]

        del a, b, rho_mask




        # 4.1.3 convergence criteria
        Jt = Einsum.transpose(model_new['J'])
        Jt_w_yyp = Einsum.chained_matmul(Jt, self.weights, y_yp_new)

        convergence_1 = xp.abs(Jt_w_yyp).max(axis=-2, keepdims=True) < LMFit.epsilon_1

        convergence_2 = xp.abs(h_lm / rho).max(axis=-2, keepdims=True) < LMFit.epsilon_2

        convergence_3 = (chi_2_new / (self.m_points - self.n_params + 1)) < LMFit.epsilon_3

        self.converged |= convergence_1 | convergence_2 | convergence_3

        self.fit_mask[self.converged] = True
