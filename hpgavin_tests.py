import numpy as np
import data_fitting
import matplotlib.pyplot as plt

np.random.seed(0)

n_points = 100
num_models = 20
msmnt_err = 0.5

t = np.arange(n_points)[np.newaxis, :]

weights = np.zeros((num_models, n_points, n_points))
diag = np.arange(n_points)
weights[..., [diag], [diag]] = 1 / (msmnt_err ** 2)
# weights = np.ones((num_models, n_points)) / (msmnt_err ** 2)


def example_1_model(x_data, params, **kwargs):
    p1 = params[..., [0], :]
    p2 = params[..., [1], :]
    p3 = params[..., [2], :]
    p4 = params[..., [3], :]

    m = p1 * np.exp(-1 * x_data / p2) + p3 * x_data * np.exp(-x_data / p4)

    return m


def example_2_model(x_data, params, **kwargs):
    p1 = params[..., [0], :]
    p2 = params[..., [1], :]
    p3 = params[..., [2], :]
    p4 = params[..., [3], :]

    m = (p1 * (x_data / np.max(x_data)) + p2 * np.power((x_data / np.max(x_data)), 2) +
         p3 * np.power((x_data / np.max(x_data)), 3) + p4 * np.power((x_data / np.max(x_data)), 4))

    return m


def example_3_model(x_data, params, **kwargs):
    p1 = params[..., [0], :]
    p2 = params[..., [1], :]
    p3 = params[..., [2], :]
    p4 = params[..., [3], :]

    m = p1 * np.exp(-1 * x_data / p2) + p3 * np.sin(x_data / p4)

    return m


def make_parameter_grids(p_true):
    p_true = p_true[np.newaxis, :]
    grid = np.repeat(p_true, num_models, axis=0)
    step = (p_true / num_models * 2) * (1 + np.arange(num_models)[:, np.newaxis])

    p1_grid = np.copy(grid)
    p1_grid[:, 0] = step[:, 0]

    p2_grid = np.copy(grid)
    p2_grid[:, 1] = step[:, 1]

    p3_grid = np.copy(grid)
    p3_grid[:, 2] = step[:, 2]

    p4_grid = np.copy(grid)
    p4_grid[:, 3] = step[:, 3]

    p_grids = {}
    p_grids['p_true'] = p_true
    p_grids['p1_grid'] = p1_grid
    p_grids['p2_grid'] = p2_grid
    p_grids['p3_grid'] = p3_grid
    p_grids['p4_grid'] = p4_grid
    p_grids['full_grid'] = step

    return p_grids


def make_chi_2(p_grids, fn):
    m0 = fn(t, p_grids['p_true'][..., np.newaxis])
    m1 = fn(t, p_grids['p1_grid'][..., np.newaxis])
    m2 = fn(t, p_grids['p2_grid'][..., np.newaxis])
    m3 = fn(t, p_grids['p3_grid'][..., np.newaxis])
    m4 = fn(t, p_grids['p4_grid'][..., np.newaxis])

    n_params = 4
    c1 = np.sum(np.power((m0 - m1), 2), axis=-2, keepdims=True) / (n_points - n_params)
    c2 = np.sum(np.power((m0 - m2), 2), axis=-2, keepdims=True) / (n_points - n_params)
    c3 = np.sum(np.power((m0 - m3), 2), axis=-2, keepdims=True) / (n_points - n_params)
    c4 = np.sum(np.power((m0 - m4), 2), axis=-2, keepdims=True) / (n_points - n_params)

    c = {}
    c['c1'] = c1
    c['c2'] = c2
    c['c3'] = c3
    c['c4'] = c4

    return c


def do_example_1():
    p_true = np.array([20.0, 10.0, 1.0, 50.0])

    grids = make_parameter_grids(p_true)

    chi_2 = make_chi_2(grids, example_1_model)

    y_true = example_1_model(t[..., np.newaxis], grids['p_true'][..., np.newaxis])
    y_data = y_true + (msmnt_err * np.random.randn(*y_true.shape))
    y_data = y_data.reshape(y_data.shape[:-1])

    fitted_results = data_fitting.LMFit(example_1_model, t, y_data, grids['full_grid'], weights, full_covar=True)
    return fitted_results, y_data


def do_example_2():
    p_true = np.array([20.0, -24.0, 30.0, -40.0])

    grids = make_parameter_grids(p_true)

    chi_2 = make_chi_2(grids, example_2_model)

    y_true = example_2_model(t[..., np.newaxis], grids['p_true'][..., np.newaxis])
    y_data = y_true + (msmnt_err * np.random.randn(*y_true.shape))
    y_data = y_data.reshape(y_data.shape[:-1])

    fitted_results = data_fitting.LMFit(example_2_model, t, y_data, grids['full_grid'], weights, full_covar=True)
    return fitted_results, y_data


def do_example_3():
    p_true = np.array([6.3, 20.78, 1.14, 5.2])

    grids = make_parameter_grids(p_true)

    chi_2 = make_chi_2(grids, example_3_model)

    y_true = example_3_model(t[..., np.newaxis], grids['p_true'][..., np.newaxis])
    y_data = y_true + (msmnt_err * np.random.randn(*y_true.shape))
    y_data = y_data.reshape(y_data.shape[:-1])

    fitted_results = data_fitting.LMFit(example_3_model, t, y_data, grids['full_grid'], weights, full_covar=True, verbose=True)
    return fitted_results, y_data


fit1, y1 = do_example_1()
fit2, y2 = do_example_2()
fit3, y3 = do_example_3()

t = t[0, :]
y1 = y1[0, :]
y2 = y2[0, :]
y3 = y3[0, :]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
ax1.scatter(t, y1, marker='.', label='True')
ax2.scatter(t, y2, marker='.', label='True')
ax3.scatter(t, y3, marker='.', label='True')

best_1 = np.argmin(fit1.chi_2, axis=0)
worst_1 = np.argmax(fit1.chi_2, axis=0)
best_2 = np.argmin(fit2.chi_2, axis=0)
worst_2 = np.argmax(fit2.chi_2, axis=0)
best_3 = np.argmin(fit3.chi_2, axis=0)
worst_3 = np.argmax(fit3.chi_2, axis=0)

y1_fit = example_1_model(t, fit1.params[best_1])
y2_fit = example_2_model(t, fit2.params[best_2])
y3_fit = example_3_model(t, fit3.params[best_3])

print('fit1 - Best: {}\tWorst: {}'.format(fit1.chi_2[best_1], fit1.chi_2[worst_1]))
print('fit2 - Best: {}\tWorst: {}'.format(fit2.chi_2[best_2], fit2.chi_2[worst_2]))
print('fit3 - Best: {}\tWorst: {}'.format(fit3.chi_2[best_3], fit3.chi_2[worst_3]))

ax1.plot(t, y1_fit[0], label='Fitted')
ax1.plot(t, example_1_model(t, fit1.params[worst_1])[0], label='Worst')
ax2.plot(t, y2_fit[0], label='Fitted')
ax2.plot(t, example_2_model(t, fit2.params[worst_2])[0], label='Worst')
ax3.plot(t, y3_fit[0], label='Fitted')
ax3.plot(t, example_3_model(t, fit3.params[worst_3])[0], label='Worst')

ax1.legend()
ax2.legend()
ax3.legend()

plt.show()
