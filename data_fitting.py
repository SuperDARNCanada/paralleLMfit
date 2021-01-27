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


    def __init__(self, data, model_dict, weights, params_dict, lm_step_shape, num_points=None):
        super(LMFit, self).__init__()

        xp = get_backend(weights)

        self.lm_step = xp.ones(lm_step_shape) * LMFit.L_0

        self.converged = xp.full(lm_step_shape, False)

        # Using matrix operations, it's not possible to selectively perform fits. Masks are used
        # to avoid fitting issues when fits have finished or failed.
        self.fit_mask = xp.full(lm_step_shape, False)

        self.n_params = len(params_dict.keys())

        if num_points is not None:
            tmp_points = xp.array(xp.broadcast_to(num_points, lm_step_shape))
            self.fit_mask[tmp_points<=0] = True

            tmp_points[tmp_points<=0] = self.n_params
            self.m_points = tmp_points
        else:
            self.m_points = data.shape[-1]

        n_iters = self.n_params * LMFit.iter_multiplier

        self.cov_mat = None
        i = 0

        prev_num_stopped_fits = 0
        while True:

            if i == n_iters:
                break

            print("Running step: ", i)
            self.levenburg_marquardt_iteration(data, model_dict, weights, params_dict)

            if i == 0:
                prev_num_stopped_fits = xp.count_nonzero(self.fit_mask)
            else:
                num_stopped_fits = xp.count_nonzero(self.fit_mask)
                change = 1 - (prev_num_stopped_fits / num_stopped_fits)
                if change < LMFit.epsilon_5:
                    break
                prev_num_stopped_fits = num_stopped_fits

            i += 1

        self.compute_cov_mat(model_dict, weights, params_dict)
        self.fitted_params = params_dict


    def compute_chi2(self, y_yp, weights):
        """
        Calculates the chi_2 values using eqn #2 from the reference document.

        :param      y_yp:     data - model
        :type       y_yp:
        :param      weights:  The weighted residuals.
        :type       weights:  { type_description }
        """

        xp = get_backend(weights)

        y_ypt = Einsum.transpose(y_yp)
        chi_2 = Einsum.chained_matmul(y_ypt, weights[:,:,xp.newaxis,:,:], y_yp)
        chi_2 = Einsum.reduce_dimensions(chi_2)

        return chi_2

    def get_model_and_J(self, model_dict, params_dict):
        """
        Evaluates the model with new parameters. If the user supplied function does not return
        a Jacobian along with the evaluated function, then the central finite differences method
        in 4.1.2 is used. Currently each iteration will use finite difference, but future
        improvements will implement the optimizations outlined.

        :param      model_dict:   model_fn : Reference to the function that yields the model and
                                  residuals
                                  args : possible args used by the model_fn
        :type       model_dict:   dict
        :param      params_dict:  Has a key for each param.
                                  'param' : {'values','min','max'}
        :type       params_dict:  dict

        :returns:   The model and Jacobian.
        :rtype:     dict
        """

        model = model_dict['model_fn'](params_dict, **model_dict['args'])

        xp = get_backend(model['model'])

        if 'J' not in model:
            J_shape = model['model'].shape + (self.n_params,)
            J = xp.empty(J_shape, dtype=model['model'].dtype)

            for i, k in enumerate(params_dict.keys()):
                tmp_params_upper = copy.deepcopy(params_dict)
                tmp_params_lower = copy.deepcopy(params_dict)

                dp = self.delta_p * (1 + xp.abs(params_dict[k]['values']))

                tmp_params_upper[k]['values'] += dp
                tmp_params_lower[k]['values'] -= dp

                m1 = model_dict['model_fn'](tmp_params_upper, **model_dict['args'])
                m2 = model_dict['model_fn'](tmp_params_lower, **model_dict['args'])

                J[...,i] = (m1['model'] - m2['model']) / (2 * dp)

            model['J'] = J

        return model

    def compute_cov_mat(self, model_dict, weights, params_dict):
        """
        Calculates the covariance matrix.

        :param      model_dict:   model_fn : Reference to the function that yields the model and
                                  residuals
                                  args : possible args used by the model_fn
        :type       model_dict:   dict
        :param      weights:      The weighted residuals.
        :type       weights:      ndarray
        :param      params_dict:  Has a key for each param.
                                  'param' : {'values','min','max'}
        :type       params_dict:  dict
        """

        xp = get_backend(weights)

        model = self.get_model_and_J(model_dict, params_dict)

        Jt = Einsum.transpose(model['J'])
        Jt_w = Einsum.matmul(Jt, weights[...,xp.newaxis,:,:])
        Jt_w_J = Einsum.matmul(Jt_w, model['J'])

        diag = xp.arange(Jt_w_J.shape[-1])

        stopped_grad = xp.full(Jt_w_J.shape, False)
        stopped_grad[...,diag,diag] = ~self.converged
        Jt_w_J[stopped_grad] = 1.0

        self.cov_mat = xp.linalg.inv(Jt_w_J)

    def levenburg_marquardt_iteration(self, data, model_dict, weights, params_dict):
        """
        Performs a single step of the LM algorithm.

        :param      data:         The measured data points.
        :type       data:         { type_description }
        :param      model_dict:   model_fn : Reference to the function that yields the model and
                                  residuals
                                  args : possible args used by the model_fn
        :type       model_dict:   dict
        :param      weights:      The weighted residuals.
        :type       weights:      ndarray
        :param      params_dict:  Has a key for each param.
                                  'param' : {'values','min','max'}
        :type       params_dict:  dict
        """

        xp = get_backend(weights)

        model = self.get_model_and_J(model_dict, params_dict)

        y_yp = (data[...,xp.newaxis,:] - model['model'])[...,xp.newaxis]
        chi_2 = self.compute_chi2(y_yp, weights)

        # Eqn #13
        Jt = Einsum.transpose(model['J'])
        Jt_w = Einsum.matmul(Jt, weights[...,xp.newaxis,:,:])
        Jt_w_J = Einsum.matmul(Jt_w, model['J'])

        diag = xp.arange(Jt_w_J.shape[-1])

        lm_Jt_w_J_diag = xp.zeros(Jt_w_J.shape)
        lm_Jt_w_J_diag[...,diag,diag] = (self.lm_step * Jt_w_J[...,diag,diag])

        grad = Jt_w_J + lm_Jt_w_J_diag


        # When fits converge/diverge such that the gradients tend to 0, this stops the matrix
        # from becoming singular.

        invertable = xp.linalg.cond(grad) < 1/xp.finfo(grad.dtype).eps
        self.fit_mask[~invertable] = True

        stopped_grad = xp.full(grad.shape, False)
        stopped_grad[...,diag,diag] = self.fit_mask
        grad[stopped_grad] = 1.0


        Jt_w_Jinv = xp.linalg.inv(grad)
        Jt_w_yyp = Einsum.matmul(Jt_w, y_yp)
        h_lm = Einsum.matmul(Jt_w_Jinv, Jt_w_yyp)

        del model, y_yp, Jt, Jt_w, Jt_w_J, diag, stopped_grad, grad, Jt_w_Jinv


        # Update the parameters and check whether they are in bounds. This algorithm is primarily
        # meant to fit oscillating functions so we stop fitting once a parameter goes out of bounds.
        # It is very likely that the algorithm will not be able to properly step in the chi_2 space
        # and will never converge.
        tmp_params = copy.copy(params_dict)
        for i, key in enumerate(tmp_params.keys()):
            tmp_params[key] = {'values' : params_dict[key]['values'] + h_lm[...,i,:]}

            if params_dict[key]['min'] is not None:
                d_min = tmp_params[key]['values'] < params_dict[key]['min']

                if isinstance(params_dict[key]['min'], xp.ndarray):
                    tmp_params[key]['values'][d_min] = params_dict[key]['min'][d_min]
                else:
                    tmp_params[key]['values'][d_min] = params_dict[key]['min']


                self.fit_mask[d_min] = True

            if params_dict[key]['max'] is not None:
                d_max = tmp_params[key]['values'] > params_dict[key]['max']

                if isinstance(params_dict[key]['max'], xp.ndarray):
                    tmp_params[key]['values'][d_max] = params_dict[key]['max'][d_max]
                else:
                    tmp_params[key]['values'][d_max] = params_dict[key]['max']

                self.fit_mask[d_max] = True




        # Eqn #16, calculating rho
        model_new = self.get_model_and_J(model_dict, tmp_params)

        y_yp_new = (data[...,xp.newaxis,:] - model_new['model'])[...,xp.newaxis]
        chi_2_new = self.compute_chi2(y_yp_new, weights)
        self.chi_2 = chi_2_new

        rho_numerator = chi_2 - chi_2_new
        rho_denominator = Einsum.matmul(Einsum.transpose(h_lm),
                                        (Einsum.matmul(lm_Jt_w_J_diag, h_lm) + Jt_w_yyp))
        rho_denominator = Einsum.reduce_dimensions(rho_denominator)

        rho = rho_numerator / rho_denominator

        # When fitting stops, this will stop div by zero later.
        self.fit_mask[rho == 0.0] = True
        rho[self.fit_mask] = 1.0


        del lm_Jt_w_J_diag, Jt_w_yyp, chi_2, rho_numerator, rho_denominator,



        # using 4.1.1 method #1 for the update
        a = xp.maximum(self.lm_step / LMFit.L_decrease, LMFit.L_min)
        a[rho<=LMFit.epsilon_4] = 0.0

        b = xp.minimum(self.lm_step * LMFit.L_increase, LMFit.L_max)
        b[rho>LMFit.epsilon_4] = 0.0

        self.lm_step = a + b
        rho_mask = (rho > LMFit.epsilon_4) & ~self.fit_mask

        for i, key in enumerate(params_dict.keys()):
            params_dict[key]['values'][rho_mask] = tmp_params[key]['values'][rho_mask]


        del a, b, rho_mask


        # 4.1.3 convergence criteria
        Jt = Einsum.transpose(model_new['J'])
        Jt_w_yyp = Einsum.chained_matmul(Jt, weights[...,xp.newaxis,:,:], y_yp_new)

        convergence_1 = xp.abs(Jt_w_yyp).max(axis=3) < LMFit.epsilon_1

        convergence_2 = xp.abs(h_lm / rho[...,xp.newaxis,:]).max(axis=3) < LMFit.epsilon_2

        convergence_3 = (chi_2_new / (self.m_points - self.n_params + 1)) < LMFit.epsilon_3

        self.converged |= convergence_1 | convergence_2 | convergence_3
        self.fit_mask[self.converged] = True
