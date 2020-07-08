import numpy as np
import copy
import sys

def einsum_transpose(ndarray):
    return np.einsum('...lm->...ml', ndarray)

def einsum_matmul(ndarray1, ndarray2):
    return np.einsum('...ij,...jk->...ik', ndarray1, ndarray2)

def einsum_chained_matmul(ndarray1, ndarray2, ndarray3):
    return np.einsum('...ij,...jk,...km->...im', ndarray1, ndarray2, ndarray3)

def einsum_reduce_dimensions(ndarray):
    return  np.einsum('...lm->...l', ndarray)

def compute_chi2(y_yp, weights):

    y_ypt = einsum_transpose(y_yp)
    chi_2 = einsum_chained_matmul(y_ypt, weights[:,:,np.newaxis,:,:], y_yp)
    chi_2 = einsum_reduce_dimensions(chi_2)

    return chi_2

def levenburg_marquardt_update(acf, model_constants, weights, fitted_params, lm_step, converged, diverged):

    e1 = 1e-3
    e2 = 1e-3
    e3 = 1e-1
    e4 = 1e-1

    fit_mask = (converged | diverged)
    fit_mask_inv = ~fit_mask


    model = compute_model_and_derivatives(model_constants, **fitted_params)
    y_yp = (acf[:,:,np.newaxis,:] - model['model'])[...,np.newaxis]
    chi_2 = compute_chi2(y_yp, weights)

    Jt = einsum_transpose(model['J'])
    Jt_w = einsum_matmul(Jt, weights[:,:,np.newaxis,:,:])
    Jt_w_J = einsum_matmul(Jt_w, model['J'])

    diag = np.arange(Jt_w_J.shape[-1])

    lm_Jt_w_J_diag = np.zeros(Jt_w_J.shape)
    lm_Jt_w_J_diag[...,diag,diag] = (lm_step * Jt_w_J[...,diag,diag])


    grad = Jt_w_J + lm_Jt_w_J_diag
    converged_grad = np.full(grad.shape, False)
    converged_grad[...,diag,diag] = fit_mask
    grad[converged_grad] = 1.0


    Jt_w_Jinv = np.linalg.inv(grad)

    Jt_w_yyp = einsum_matmul(Jt_w, y_yp)

    h_lm = einsum_matmul(Jt_w_Jinv, Jt_w_yyp)

    tmp_params = copy.copy(fitted_params)

    tmp_params['p0'] = fitted_params['p0'] + h_lm[:,:,:,0]
    tmp_params['W'] = fitted_params['W'] + h_lm[:,:,:,1]
    tmp_params['V'] = fitted_params['V'] + h_lm[:,:,:,2]


    d1 = tmp_params['p0'] < 0.0
    d2 = tmp_params['W'] < -200
    d3 = tmp_params['V'] > 3000
    d4 = tmp_params['V'] < -3000

    tmp_params['p0'][d1] = 0.0
    tmp_params['W'][d2] = -200
    tmp_params['V'][d3] = 3000
    tmp_params['V'][d4] = -3000
    diverged |= d1 | d2 | d3 | d4

    fit_mask = (converged | diverged)
    fit_mask_inv = ~fit_mask

    model_new = compute_model_and_derivatives(model_constants, **tmp_params)
    y_yp_new = (acf[:,:,np.newaxis,:] - model_new['model'])[...,np.newaxis]
    chi_2_new = compute_chi2(y_yp_new, weights)


    rho_numerator = chi_2 - chi_2_new

    rho_denominator = einsum_matmul(einsum_transpose(h_lm),
                                    (einsum_matmul(lm_Jt_w_J_diag, h_lm) + Jt_w_yyp))
    rho_denominator = einsum_reduce_dimensions(rho_denominator)

    rho = rho_numerator/rho_denominator

    rho[fit_mask] = 1.0

    L_increase = 11
    L_decrease = 9

    a = np.maximum(lm_step/L_decrease, 1e-7)
    a[rho<=e4] = 0.0

    b = np.minimum(lm_step*L_increase, 1e7)
    b[rho>e4] = 0.0

    lm_step_new = a + b


    rho_mask = (rho>e4) & fit_mask_inv


    fitted_params['p0'][rho_mask] = tmp_params['p0'][rho_mask]
    fitted_params['W'][rho_mask] = tmp_params['W'][rho_mask]
    fitted_params['V'][rho_mask] = tmp_params['V'][rho_mask]




    Jt = einsum_transpose(model_new['J'])
    Jt_w_yyp = einsum_chained_matmul(Jt, weights[:,:,np.newaxis,:,:], y_yp_new)

    convergence_1 = np.abs(Jt_w_yyp).max(axis=3) < e1

    convergence_2 = np.abs(h_lm/rho[...,np.newaxis,:]).max(axis=3) < e2

    m = model_new['model'].shape[-1]
    n = model_new['J'].shape[-1]

    convergence_3 = (chi_2_new/(m - n + 1)) < e3

    converged |= convergence_1 | convergence_2 | convergence_3

    return lm_step_new



def compute_model_and_derivatives(model_constants, p0, W, V):
    model_dict = {}

    mc_W = model_constants['W']
    mc_V = model_constants['V']

    def calculate_model():
        model = p0 * np.exp(mc_W * W) * np.exp(mc_V * V)
        return model

    model = calculate_model()
    J = np.repeat(model[:,:,:,:,np.newaxis], 3, axis=4)

    def compute_J():
        J[:,:,:,:,0] /= p0
        J[:,:,:,:,1] *= mc_W
        J[:,:,:,:,2] *= mc_V

        return J

    model_dict['model'] = np.concatenate((model.real, model.imag), axis=-1)

    J_model = compute_J()
    J_model = np.concatenate((J_model.real, J_model.imag), axis=-2)
    model_dict['J'] = J_model

    return model_dict

def calculate_gauss_newton_update(acf, model, weights):
    y_yp = (acf[:,:,np.newaxis,:] - model['model'])[...,np.newaxis]

    Jt = einsum_transpose(model['J'])
    Jt_w = einsum_matmul(Jt, weights[:,:,np.newaxis,:,:])
    Jt_w_J = einsum_matmul(Jt_w, model['J'])

    diag = np.arange(Jt_w_J.shape[-1])
    Jt_w_J[...,diag,diag] += 1e-6
    Jt_w_Jinv = np.linalg.inv(Jt_w_J)

    Jt_w_yyp = einsum_matmul(Jt_w, y_yp)

    h_gn = einsum_matmul(Jt_w_Jinv, Jt_w_yyp)

    return h_gn

def fit_data(params, model_constants, acf, weights):

    fitted_params = params

    shape = (acf.shape[0], acf.shape[1], fitted_params['V'].shape[2], 1)
    lm_step = np.ones(shape) * 1e-2

    converged = np.full(shape, False)
    diverged = np.full(shape, False)
    i = 0
    while (i<30) and not np.all(converged | diverged):
        print("Running step: ", i)
        lm_step = levenburg_marquardt_update(acf, model_constants, weights, fitted_params, lm_step, converged, diverged)

        print('p0', fitted_params['p0'][1,20,:,0])
        print('W', fitted_params['W'][1,20,:,0])
        print('V', fitted_params['V'][1,20,:,0])
        print('fit_status', (converged | diverged)[1,20,:,0])

        i += 1

    sys.exit(1)
    return fitted_params