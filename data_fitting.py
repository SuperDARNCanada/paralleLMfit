import numpy as np

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
    chi_2 = einsum_chained_matmul(y_ypt, weights[:,:,np.newaxis,:,:], y_ypt)
    chi_2 = einsum_reduce_dimensions(chi_2)

    return chi_2

def levenburg_marquardt_update(acf, model_constants, weights, fitted_params, lm_step):

    model = compute_model_and_derivatives(model_constants, **fitted_params)
    y_yp = (acf[:,:,np.newaxis,:] - model['model'])[...,np.newaxis]
    chi_2 = compute_chi2(y_yp, weights)

    Jt = einsum_transpose(model['J'])
    Jt_w = einsum_matmul(Jt, weights[:,:,np.newaxis,:,:])
    Jt_w_J = einsum_matmul(Jt_w, model['J'])

    diag = np.arange(Jt_w_J.shape[-1])
    Jt_w_J_diag = Jt_w_J[...,diag,diag]

    lm_Jt_w_J_diag = np.zeros(Jt_w_J.shape)
    lm_Jt_w_J_diag[...,diag,diag] = (lm_step + Jt_w_J[...,diag,diag])


    Jt_w_Jinv = np.linalg.inv(Jt_w_J + lm_Jt_w_J_diag)

    Jt_w_yyp = einsum_matmul(Jt_w, y_yp)

    h_lm = einsum_matmul(Jt_w_Jinv, Jt_w_yyp)
    #print(h_lm[0,0,0])

    # tmp_p0 = fitted_params['p0'] + h_lm[:,:,:,0]
    # tmp_W = fitted_params['W'] + h_lm[:,:,:,1]
    # tmp_V = = fitted_params['V'] + h_lm[:,:,:,2]

    # tmp_fitted_params = {'p0' : tmp_p0, 'W' : tmp_W, 'V' : tmp_V}

    fitted_params['p0'] = fitted_params['p0'] + h_lm[:,:,:,0]
    fitted_params['W'] = fitted_params['W'] + h_lm[:,:,:,1]
    fitted_params['V'] = fitted_params['V'] + h_lm[:,:,:,2]

    model_new = compute_model_and_derivatives(model_constants, **fitted_params)
    y_yp_new = (acf[:,:,np.newaxis,:] - model_new['model'])[...,np.newaxis]
    chi_2_new = compute_chi2(y_yp_new, weights)

    rho_numerator = chi_2 - chi_2_new

    rho_denominator = einsum_matmul(einsum_transpose(h_lm),
                                    (einsum_matmul(lm_Jt_w_J_diag, h_lm) + Jt_w_yyp))
    rho_denominator = einsum_reduce_dimensions(rho_denominator)

    rho = rho_numerator/rho_denominator
    #print(rho.shape, h_lm.shape)

    L_increase = 11
    L_decrease = 9

    a = np.maximum(lm_step/L_decrease, 1e-7)
    a[rho<=1e-1] = 0.0
    # tmp = a
    # tmp[rho<=1e-1] = 0.0
    # print(tmp[0,0])


    #print('a', a[0,0])

    b = np.minimum(lm_step*L_increase, 1e7)
    b[rho>1e-1] = 0.0
    # print(b[0,0])

    #print('b', b[0,0])

    lm_step_new = a + b
    # lm_step_new = np.zeros(lm_step.shape)
    # lm_step_new[rho>1e-1] = a[rho>1e-1]
    # lm_step_new[rho<=1e-1] = b[rho<=1e-1]

    #print(lm_step_new[0,0])

    tmp_h_lm = np.where(rho[...,np.newaxis]<=1e-1, h_lm, 0.0)
    #print(tmp_h_lm[0,0,:,0])

    fitted_params['p0'] = fitted_params['p0'] - tmp_h_lm[:,:,:,0]
    fitted_params['W'] = fitted_params['W'] - tmp_h_lm[:,:,:,1]
    fitted_params['V'] = fitted_params['V'] - tmp_h_lm[:,:,:,2]

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

    model_dict['model'] =np.concatenate((model.real, model.imag), axis=-1)

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
    # model = compute_model_and_derivatives(model_constants, **fitted_params)
    # y_yp = (acf[:,:,np.newaxis,:] - model['model'])[...,np.newaxis]
    # chi_2 = compute_chi2(y_yp, weights)

    shape = (acf.shape[0], acf.shape[1], fitted_params['V'].shape[2], 1)
    lm_step = np.ones(shape) * 1e-4
    print(lm_step.shape)

    for i in range(50):
        lm_step = levenburg_marquardt_update(acf, model_constants, weights, fitted_params, lm_step)
        #model = compute_model_and_derivatives(model_constants, **fitted_params)

        # h_gn = calculate_gauss_newton_update(acf, model, weights)
        #print(lm_step[0,0,0])

        print('p0', fitted_params['p0'][0,0,:,0])
        print('W', fitted_params['W'][0,0,:,0])
        print('V', fitted_params['V'][0,0,:,0])

        # fitted_params['p0'] = fitted_params['p0'] + h_gn[:,:,:,0]
        # fitted_params['W'] = fitted_params['W'] + h_gn[:,:,:,1]
        # fitted_params['V'] = fitted_params['V'] + h_gn[:,:,:,2]

    sys.exit(1)
    return fitted_params