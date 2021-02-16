import numpy as np
import data_fitting

np.random.seed(0)

n_points = 100
num_models = 20
msmnt_err = 0.5

t = np.arange(n_points)[np.newaxis,:,np.newaxis]

weights = np.zeros((num_models,n_points,n_points))
diag = np.arange(n_points)
weights[...,[diag],[diag]] = 1/(msmnt_err**2)

def example_1_model(x_data, params, **kwargs):
    p1 = params[...,[0],:]
    p2 = params[...,[1],:]
    p3 = params[...,[2],:]
    p4 = params[...,[3],:]

    m = (p1 * (x_data / np.max(x_data)) + p2 * np.power((x_data / np.max(x_data)), 2) +
        p3 * np.power((x_data / np.max(x_data)), 3) + p4 * np.power((x_data / np.max(x_data)), 4))

    return m

def example_2_model(x_data, params, **kwargs):
    p1 = params[...,[0],:]
    p2 = params[...,[1],:]
    p3 = params[...,[2],:]
    p4 = params[...,[3],:]

    m = p1 * np.exp(-1*x_data/p2) + p3*np.sin(x_data/p4)

    return m

def example_3_model(x_data, params, **kwargs):
    p1 = params[...,[0],:]
    p2 = params[...,[1],:]
    p3 = params[...,[2],:]
    p4 = params[...,[3],:]

    m = p1 * np.exp(-1*x_data/p2) + p3*np.sin(x_data/p4)

    return m

def make_paramater_grids(p_true):
    p_true = p_true[np.newaxis,:,np.newaxis]
    grid = np.repeat(p_true, num_models, axis=0)
    step = (p_true / num_models * 2) * (1+np.arange(num_models)[:,np.newaxis,np.newaxis])

    p1_grid = np.copy(grid)
    p1_grid[:,0] = step[:,0]

    p2_grid = np.copy(grid)
    p2_grid[:,1] = step[:,1]

    p3_grid = np.copy(grid)
    p3_grid[:,2] = step[:,2]

    p4_grid = np.copy(grid)
    p4_grid[:,3] = step[:,3]

    p_grids = {}
    p_grids['p_true'] = p_true
    p_grids['p1_grid'] = p1_grid
    p_grids['p2_grid'] = p2_grid
    p_grids['p3_grid'] = p3_grid
    p_grids['p4_grid'] = p4_grid
    p_grids['full_grid'] = step

    return p_grids

def make_chi_2(p_grids, fn):

    m0 = fn(t, p_grids['p_true'])
    m1 = fn(t, p_grids['p1_grid'])
    m2 = fn(t, p_grids['p2_grid'])
    m3 = fn(t, p_grids['p3_grid'])
    m4 = fn(t, p_grids['p4_grid'])

    n_params = 4
    c1 = np.sum(np.power((m0 - m1), 2), axis=-2, keepdims=True)/(n_points - n_params)
    c2 = np.sum(np.power((m0 - m2), 2), axis=-2, keepdims=True)/(n_points - n_params)
    c3 = np.sum(np.power((m0 - m3), 2), axis=-2, keepdims=True)/(n_points - n_params)
    c4 = np.sum(np.power((m0 - m4), 2), axis=-2, keepdims=True)/(n_points - n_params)

    c = {}
    c['c1'] = c1
    c['c2'] = c2
    c['c3'] = c3
    c['c4'] = c4

    return c

def do_example_1():
    p_true = np.array([20.0,10.0,1.0,50.0])

    grids = make_paramater_grids(p_true)

    chi_2 = make_chi_2(grids, example_1_model)

    y_true = example_1_model(t, grids['p_true'])
    y_data = y_true + (msmnt_err * np.random.randn(*y_true.shape))

    fitted_results = data_fitting.LMFit(example_1_model, t, y_data, grids['full_grid'], weights)

def do_example_2():
    p_true = np.array([20.0,-24.0,30.0,-40.0])

    grids = make_paramater_grids(p_true)

    chi_2 = make_chi_2(grids, example_2_model)

    y_true = example_2_model(t, grids['p_true'])
    y_data = y_true + (msmnt_err * np.random.randn(*y_true.shape))

    fitted_results = data_fitting.LMFit(example_2_model, t, y_data, grids['full_grid'], weights)


def do_example_3():
    p_true = np.array([6.0,20.0,1.0,5.0])

    grids = make_paramater_grids(p_true)

    chi_2 = make_chi_2(grids, example_3_model)

    y_true = example_3_model(t, grids['p_true'])
    y_data = y_true + (msmnt_err * np.random.randn(*y_true.shape))

    fitted_results = data_fitting.LMFit(example_3_model, t, y_data, grids['full_grid'], weights)





do_example_1()
do_example_2()
do_example_3()









