import numpy as np
import copy
import sys

class Einsum(object):
    """Einsum helper functions."""

    @staticmethod
    def transpose(ndarray):
        transpose_definition = '...lm->...ml'
        return np.einsum(transpose_definition, ndarray)

    @staticmethod
    def matmul(ndarray1, ndarray2):
        matmul_definition = '...ij,...jk->...ik'
        return np.einsum(matmul_definition, ndarray1, ndarray2)

    @staticmethod
    def chained_matmul(ndarray1, ndarray2, ndarray3):
        chained_definition = '...ij,...jk,...km->...im'
        return np.einsum(chained_definition , ndarray1, ndarray2, ndarray3)

    @staticmethod
    def reduce_dimensions(ndarray):
        reduce_definition = '...lm->...l'
        return np.einsum(reduce_definition, ndarray)


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

    L_0 = 1e-2
    L_increase = 11
    L_decrease = 9
    L_max = 1e7
    L_min = 1e-7

    iter_multiplier = 10


    def __init__(self, data, model_dict, weights, params_dict, lm_step_shape):
        super(LMFit, self).__init__()

        self.lm_step = np.ones(lm_step_shape) * LMFit.L_0

        self.converged = np.full(lm_step_shape, False)
        self.diverged = np.full(lm_step_shape, False)

        n_params = len(params_dict.keys())
        n_iters = n_params * LMFit.iter_multiplier

        i = 0
        while (i<n_iters) and not np.all(self.converged | self.diverged):

            print("Running step: ", i)
            self.levenburg_marquardt_iteration(data, model_dict, weights, params_dict)

            print('p0', params_dict['p0']['values'][0,0,:,0])
            print('W', params_dict['W']['values'][0,0,:,0])
            print('V', params_dict['V']['values'][0,0,:,0])
            print('fit_status', (self.converged | self.diverged)[0,0,:,0])

            i += 1


    def compute_chi2(self, y_yp, weights):
        """
        Calculates the chi_2 values using eqn #2 from the reference document.

        :param      y_yp:     data - model
        :type       y_yp:
        :param      weights:  The weighted residuals.
        :type       weights:  { type_description }
        """

        y_ypt = Einsum.transpose(y_yp)
        chi_2 = Einsum.chained_matmul(y_ypt, weights[:,:,np.newaxis,:,:], y_yp)
        chi_2 = Einsum.reduce_dimensions(chi_2)

        return chi_2

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


        # Using matrix operations, it's not possible to selectively perform fits. Masks are used
        # to avoid fitting issues when fits have finished.
        fit_mask = (self.converged | self.diverged)
        fit_mask_inv = ~fit_mask


        model = model_dict['model_fn'](params_dict, **model_dict['args'])

        y_yp = (data[...,np.newaxis,:] - model['model'])[...,np.newaxis]
        chi_2 = self.compute_chi2(y_yp, weights)


        # Eqn #13
        Jt = Einsum.transpose(model['J'])
        Jt_w = Einsum.matmul(Jt, weights[...,np.newaxis,:,:])
        Jt_w_J = Einsum.matmul(Jt_w, model['J'])

        diag = np.arange(Jt_w_J.shape[-1])

        lm_Jt_w_J_diag = np.zeros(Jt_w_J.shape)
        lm_Jt_w_J_diag[...,diag,diag] = (self.lm_step * Jt_w_J[...,diag,diag])

        grad = Jt_w_J + lm_Jt_w_J_diag

        # When fits converge/diverge such that the gradients tend to 0, this stops the matrix
        # from becoming singular.
        converged_grad = np.full(grad.shape, False)
        converged_grad[...,diag,diag] = fit_mask
        grad[converged_grad] = 1.0


        Jt_w_Jinv = np.linalg.inv(grad)
        Jt_w_yyp = Einsum.matmul(Jt_w, y_yp)
        h_lm = Einsum.matmul(Jt_w_Jinv, Jt_w_yyp)






        # Update the parameters and check whether they are in bounds. This algorithm is primarily
        # meant to fit oscillating functions so we stop fitting once a parameter goes out of bounds.
        # It is very likely that the algorithm will not be able to properly step in the chi_2 space
        # and will never converge.
        tmp_params = copy.copy(params_dict)
        for i, key in enumerate(tmp_params.keys()):
            tmp_params[key] = {'values' : params_dict[key]['values'] + h_lm[...,i]}

            if params_dict[key]['min'] is not None:
                d_min = tmp_params[key]['values'] < params_dict[key]['min']

                if isinstance(params_dict[key]['min'], np.ndarray):
                    tmp_params[key]['values'][d_min] = params_dict[key]['min'][d_min]
                else:
                    tmp_params[key]['values'][d_min] = params_dict[key]['min']


                self.diverged[d_min] = True

            if params_dict[key]['max'] is not None:
                d_max = tmp_params[key]['values'] > params_dict[key]['max']

                if isinstance(params_dict[key]['max'], np.ndarray):
                    tmp_params[key]['values'][d_max] = params_dict[key]['max'][d_max]
                else:
                    tmp_params[key]['values'][d_max] = params_dict[key]['max']


                self.diverged[d_max] = True

        # Update mask to stop fitting if diverged.
        fit_mask = (self.converged | self.diverged)
        fit_mask_inv = ~fit_mask





        # Eqn #16, calculating rho
        model_new = model_dict['model_fn'](tmp_params, **model_dict['args'])
        y_yp_new = (data[...,np.newaxis,:] - model_new['model'])[...,np.newaxis]
        chi_2_new = self.compute_chi2(y_yp_new, weights)


        rho_numerator = chi_2 - chi_2_new
        rho_denominator = Einsum.matmul(Einsum.transpose(h_lm),
                                        (Einsum.matmul(lm_Jt_w_J_diag, h_lm) + Jt_w_yyp))
        rho_denominator = Einsum.reduce_dimensions(rho_denominator)

        rho = rho_numerator/rho_denominator

        # When fitting stops, this will stop div by zero later.
        rho[fit_mask] = 1.0





        # using 4.1.1 method #1 for the update
        a = np.maximum(self.lm_step/L_decrease, LMFit.L_min)
        a[rho<=LMFit.epsilon_4] = 0.0

        b = np.minimum(self.lm_step*L_increase, LMFit.L_max)
        b[rho>LMFit.epsilon_4] = 0.0

        self.lm_step = a + b
        rho_mask = (rho>LMFit.epsilon_4) & fit_mask_inv

        for i, key in enumerate(params_dict.keys()):
            params_dict[key]['values'][rho_mask] = tmp_params[key]['values'][rho_mask]





        # 4.1.3 convergence criteria
        Jt = Einsum.transpose(model_new['J'])
        Jt_w_yyp = Einsum.chained_matmul(Jt, weights[...,np.newaxis,:,:], y_yp_new)

        convergence_1 = np.abs(Jt_w_yyp).max(axis=3) < LMFit.epsilon_1

        convergence_2 = np.abs(h_lm/rho[...,np.newaxis,:]).max(axis=3) < LMFit.epsilon_2

        m = model_new['model'].shape[-1]
        n = model_new['J'].shape[-1]

        convergence_3 = (chi_2_new/(m - n + 1)) < LMFit.epsilon_3

        self.converged |= convergence_1 | convergence_2 | convergence_3

