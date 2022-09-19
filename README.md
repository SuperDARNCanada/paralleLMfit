# ParalleLMFIT

A Python3 package for non-linear least-squares fitting using the Levenberg-Marquardt algorithm,
based on "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems" by Gavin, H. P. (2020).
Link: https://people.duke.edu/~hpgavin/ce281/lm.pdf

This package provides a flexible implementation of the Levenberg-Marquardt algorithm that is
easy to use and can offer significant speed improvements by running on a GPU.

## Usage

It is up to the user to ensure that all data arrays are sized correctly for this algorithm to work.

A simple example:

```python3
from parallelmfit import LMFit
import numpy as xp  # if desired, import cupy as xp

x = xp.arange(100)

y_true = -1*x*x + 0.5*x + 10    # ax^2 + bx + c
noise = xp.random.randn(100)    # normal distribution with mean = 1, std. dev. = 1 
y = y_true + noise

initial_params = xp.array([1, 1, 1])    # guesses for [a, b, c]

weights = xp.ones(100) / (0.5 ** 2)     # 1 / sigma^2

def model_function(x, params, **kwargs):
    """Computes the expected output (and optionally, the Jacobian)
    
    :param       x: Independent data
    :type        x: ndarray [..., num_points, 1]
    :param  params: Array of parameter values
    :type   params: ndarray [..., num_params, 1]
    :param  kwargs: Optional keyword arguments as needed (passed in from LMFit call)
    :type   kwargs: dict
    """
    model = params[[0], :]*x*x + params[[1], :]*x + params[[2], :]    # will have shape [..., num_points, 1]
    
    return model    # return (model, jacobian) if you can compute jacobian analytically
        
fit = LMFit(model_function, x, y, initial_params, weights)  # Add kwargs here if needed for model_function()

fitted_params = fit.fitted_params
chi_2 = fit.chi_2
covariance_matrix = fit.cov_mat
```