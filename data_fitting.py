import numpy as np

def einsum_transpose(ndarray):
    return np.einsum('...lm->...ml', ndarray)

def einsum_matmul(ndarray1, ndarray2):
    return np.einsum('...ij,...jk->...ik', ndarray1, ndarray2)

def einsum_chained_matmul(ndarray1, ndarray2, ndarray3):
    return np.einsum('...ij,...jk,...km->...im', ndarray1, ndarray2, ndarray3)

def einsum_reduce_dimensions(ndarray):
    return  np.einsum('...lm->...l', ndarray)

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


    model_dict['model'] =np.concatenate((model.real, model.imag), axis=-1)

    compute_J()
    J_model = np.concatenate((J.real, J.imag), axis=-2)
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

    for i in range(70):
        model = compute_model_and_derivatives(model_constants, **fitted_params)

        h_gn = calculate_gauss_newton_update(acf, model, weights)
        print(h_gn[0,0,0])

        print('p0', fitted_params['p0'][0,0,:,0])
        print('W', fitted_params['W'][0,0,:,0])
        print('V', fitted_params['V'][0,0,:,0])

        fitted_params['p0'] = fitted_params['p0'] + h_gn[:,:,:,0]
        fitted_params['W'] = fitted_params['W'] + h_gn[:,:,:,1]
        fitted_params['V'] = fitted_params['V'] + h_gn[:,:,:,2]


    return fitted_params