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

    #@jit(nopython=True)
    def calculate_model():
        model = p0 * np.exp(model_constants['real'] * W) * np.exp(model_constants['imag'] * V)
        return model

    model = calculate_model()
    J = np.repeat(model[:,:,:,:,np.newaxis], 3, axis=4)

    #@jit(nopython=True)
    def compute_J():
        J[:,:,:,:,0] /= p0
        J[:,:,:,:,1] *= model_constants['real']
        J[:,:,:,:,2] *= model_constants['imag']

        return J

    model_dict['model'] = model
    model_dict['J'] = compute_J()

    return model_dict

def calculate_gauss_newton_update(acf, model, weights):
    yreal_yp = (acf['real'] - model['model'].real)[...,np.newaxis]
    yimag_yp = (acf['imag'] - model['model'].imag)[...,np.newaxis]

    Jreal_t = einsum_transpose(model['J'].real)
    Jimag_t = einsum_transpose(model['J'].imag)

    Jreal_t_w = einsum_matmul(Jreal_t, weights['real'])
    Jimag_t_w = einsum_matmul(Jimag_t, weights['imag'])

    Jreal_t_w_Jreal = einsum_matmul(Jreal_t_w, model['J'].real)
    Jimag_t_w_Jimag = einsum_matmul(Jimag_t_w, model['J'].imag)

    Jreal_t_w_Jreal_inv = np.linalg.inv(Jreal_t_w_Jreal)
    Jimag_t_w_Jimag_inv = np.linalg.inv(Jimag_t_w_Jimag)

    Jreal_t_w_yrealyp = einsum_matmul(Jreal_t_w, yreal_yp)
    Jimag_t_w_yimagyp = einsum_matmul(Jimag_t_w, yimag_yp)

    h_gn_real = einsum_matmul(Jreal_t_w_Jreal_inv, Jreal_t_w_yrealyp)
    h_gn_imag = einsum_matmul(Jimag_t_w_Jimag_inv, Jimag_t_w_yimagyp)

    h_gn = h_gn_real + h_gn_imag

    return h_gn

def fit_data(params, model_constants, acf, weights):

    fitted_params = params
    for i in range(25):
        model = compute_model_and_derivatives(model_constants, **fitted_params)
        h_gn = calculate_gauss_newton_update(acf, model, weights)

        fitted_params['p0'] = fitted_params['p0'] + h_gn[:,:,:,0]
        print(fitted_params['p0'][0,0,0,:])
        print(h_gn[0,0,0,:])
        #fitted_params['p0'][fitted_params['p0'] < 0.0] = 0.0
        #print(np.amax(fitted_params['p0']), np.amin(fitted_params['p0']), np.mean(fitted_params['p0']))

        fitted_params['W'] = fitted_params['W'] + h_gn[:,:,:,1]
        #fitted_params['W'][fitted_params['W'] < -100.0] = -100.0
       # print(np.amax(fitted_params['W']), np.amin(fitted_params['W']), np.mean(fitted_params['W']))

        fitted_params['V'] = fitted_params['V'] + h_gn[:,:,:,2]
        #fitted_params['V'][fitted_params['V'] < -2500.0] = -2500
        #fitted_params['V'][fitted_params['V'] > 2500.0] = 2500

        #print(np.amax(fitted_params['V']), np.amin(fitted_params['V']), np.mean(fitted_params['V']))

    return fitted_params