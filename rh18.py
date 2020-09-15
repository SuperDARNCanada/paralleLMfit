import rawacf_dmap_read as rdr
import data_fitting
import sys
import time
import argparse
import deepdish as dd
import copy
from multiprocessing.pool import ThreadPool

def determine_noise(pwr0):
    """
    Determines the noise power used to weight data points.

    :param      pwr0:  The lag0 power
    :type       pwr0:  ndarray [num_records, num_ranges]

    :returns:   Noise estimate
    :rtype:     ndarray [num_records, ]
    """

    # [num_records, num_ranges]
    sorted_pwr0 = xp.sort(pwr0)

    # [num_records, 10]
    noise = xp.mean(sorted_pwr0[:,0:10], axis=1)
    return noise

def first_order_weights(pwr0, noise, clutter, nave, blanking_mask):
    """
    Ashton can write this.

    :param      pwr0:           The lag0 power.
    :type       pwr0:           ndarray [num_records, num_ranges]
    :param      noise:          The noise estimates.
    :type       noise:          ndarray [num_records, ]
    :param      clutter:        The clutter estimates.
    :type       clutter:        ndarray [num_records, num_lags+1, num_ranges]
    :param      nave:           The number of averages.
    :type       nave:           ndarray [num_records, ]
    :param      blanking_mask:  The blanking mask to indicate bad points.
    :type       blanking_mask:  ndarray [num_records, num_ranges, num_lags*2]

    :returns:   The first order weights to use during fitting.
    :rtype:     ndarray [num_records, num_ranges, num_lags*2, num_lags*2]
    """

    # [num_records, 1, num_ranges]
    # [num_records, 1, 1]
    # [num_records, num_lags+1, num_ranges]
    received_power = (pwr0[:,xp.newaxis,:] + noise[:,xp.newaxis,xp.newaxis] + clutter)

    # [num_records, num_lags+1, num_ranges]
    # [num_records, 1, 1]
    error = received_power / xp.sqrt(nave[:,xp.newaxis,xp.newaxis].astype(float))
    error = data_fitting.Einsum.transpose(error)

    # Cut alt lag 0
    # [num_records, num_ranges, num_lags+1]
    error = error[...,:-1]

    weights_shape = (error.shape[0], error.shape[1], error.shape[2] * 2, error.shape[2] * 2)
    # [num_records, num_ranges, num_lags*2, num_lags*2]
    weights = xp.zeros(weights_shape)

    diag = xp.arange(weights.shape[-1])
    diag_r = diag[0:error.shape[2]]
    diag_i = diag[error.shape[2]:]

    # Example of assigning weights
    # [r1 0 0 0 0 0]   [e1 0 0 0 0 0]
    # [0 r2 0 0 0 0]   [0 e2 0 0 0 0]
    # [0 0 r3 0 0 0] = [0 0 e3 0 0 0]
    # [0 0 0 i1 0 0]   [0 0 0 e1 0 0]
    # [0 0 0 0 i2 0]   [0 0 0 0 e2 0]
    # [0 0 0 0 0 i3]   [0 0 0 0 0 e3]
    weights[...,diag_r,diag_r] = 1.0 / error**2
    weights[...,diag_i,diag_i] = 1.0 / error**2

    # [num_records, num_ranges, num_lags*2]
    weights[...,diag,diag][blanking_mask] = 1e-20

    return weights


def calculate_samples(num_range_gates, ltab, ptab, mpinc, lagfr, smsep):
    """
    Calculate sample pairs for all the lags at each range.

    :param      num_range_gates:  The number range gates
    :type       num_range_gates:  int
    :param      ltab:             The lag table
    :type       ltab:             ndarray [num_records, num_lags, 2]
    :param      ptab:             The pulse table used to make lags.
    :type       ptab:             ndarray [num_records, num_pulses]
    :param      mpinc:            The fundamental lag spacing in time.
    :type       mpinc:            ndarray [num_records, ]
    :param      lagfr:            The lag to first range in us.
    :type       lagfr:            ndarray [num_records, ]
    :param      smsep:            The sample separation in us.
    :type       smsep:            ndarray [num_records, ]

    :returns:   Pulses converted to sample number, sample pairs for lags at each range.
    :rtype:     tuple of ndarray ([num_records, num_pulses, num_ranges],
                                  [num_records, num_lags+1, 2, num_ranges])
    """

    ranges = xp.arange(num_range_gates)
    pulses_by_ranges = xp.repeat(ptab[...,xp.newaxis], num_range_gates, axis=-1)

    # [num_records, ]
    # [num_records, ]
    ranges_per_tau = mpinc / smsep

    ranges_per_tau = ranges_per_tau[...,xp.newaxis,xp.newaxis]

    # [num_records, num_pulses, num_ranges]
    # [num_records, 1, 1]
    pulses_as_samples = (pulses_by_ranges * ranges_per_tau)

    # [num_records, ]
    # [num_records, ]
    first_range_in_samps = lagfr/smsep

    first_range_in_samps = first_range_in_samps[...,xp.newaxis,xp.newaxis]

    # [num_records, num_pulses, num_ranges]
    # [num_records, 1, 1]
    # [1, 1, num_ranges]
    # [num_records, 1, 1]
    samples_used = ((pulses_by_ranges * ranges_per_tau) +
                    ranges[xp.newaxis,xp.newaxis,:] +
                    first_range_in_samps)
    samples_used = data_fitting.Einsum.transpose(samples_used)

    def searchsorted2d(a,b):
        """https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy"""
        m,n,o = a.shape
        max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
        r = max_num*xp.arange(a.shape[0])[:,xp.newaxis,xp.newaxis]
        p = xp.searchsorted( (a+r).ravel(), (b+r).ravel() ).reshape(b.shape)
        return p - n*(xp.arange(m)[:,xp.newaxis, xp.newaxis])

    lags_pulse_idx = searchsorted2d(ptab[...,xp.newaxis], ltab)
    lags_pulse_idx = xp.repeat(lags_pulse_idx[...,xp.newaxis], num_range_gates, axis=-1)

    # [num_records, num_lags+1, 2, num_ranges]
    lags_pulse_idx = xp.einsum('...ijk->...kij', lags_pulse_idx)
    shape = lags_pulse_idx.shape
    lags_pulse_idx = lags_pulse_idx.reshape((shape[:2]) + (shape[-1] * shape[-2],))

    # [num_records, num_ranges, num_pulses]
    # [num_records, num_ranges, num_lags+1 * 2]
    samples_for_lags = xp.take_along_axis(samples_used, lags_pulse_idx, axis=-1)
    samples_for_lags = samples_for_lags.reshape(shape)

    # [num_records, num_ranges, num_lags+1, 2]
    samples_for_lags = xp.einsum('...ijk->...jki', samples_for_lags)

    return pulses_as_samples, samples_for_lags

def create_blanking_mask(num_range_gates, pulses_as_samples, samples_for_lags, data_mask,
                            num_averages):
    """
    Creates a blanking mask where bad data points are.

    :param      num_range_gates:    The number of range gates.
    :type       num_range_gates:    int
    :param      pulses_as_samples:  The pulses converted to sample number.
    :type       pulses_as_samples:  ndarray [num_records, num_pulses, num_ranges]
    :param      samples_for_lags:   The sample pairs used to create lags at each range.
    :type       samples_for_lags:   ndarray [num_records, num_lags+1, 2, num_ranges]
    :param      data_mask:          The mask indicating where valid data from the file is.
    :type       data_mask:          ndarray [num_records, num_lags+1, 2, num_ranges]
    :param      num_averages:       The number averages in each record.
    :type       num_averages:       ndarray [num_records, ]

    :returns:   Boolean mask flagging bad data points.
    :rtype:     ndarray [num_records, num_ranges, num_lags*2]
    """

    # first do tx blanked lags
    # [num_records, 1, 1, num_pulses, num_ranges]
    blanked_1 = data_fitting.Einsum.transpose(pulses_as_samples)[...,xp.newaxis,xp.newaxis,:,:]
    blanked_2 = blanked_1 + 1

    # [num_records, 1, 1, num_ranges, num_pulses]
    # [num_records, num_lags+1, 2, num_ranges, 1]
    x = blanked_1 == samples_for_lags[...,xp.newaxis]
    y = blanked_2 == samples_for_lags[...,xp.newaxis]

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    blanking_mask = xp.any(x | y, axis=-1)

    # Blank where data does not exist
    # [num_records, num_lags+1, 2, num_ranges]
    blanking_mask[~data_mask] = True

    # [num_records, ]
    blanking_mask[num_averages<=0] = True

    # [num_records, num_lags+1, 2, num_ranges]
    blanking_mask = xp.einsum('...ijk->...kji', blanking_mask)

    # cut alt lag 0
    # [num_records, num_ranges, 2, num_lags+1]
    blanking_mask = blanking_mask[...,:-1]


    blanking_mask = blanking_mask.reshape(blanking_mask.shape[0], blanking_mask.shape[1],
                                            blanking_mask.shape[2] * blanking_mask.shape[3])

    return blanking_mask


def estimate_max_self_clutter(num_range_gates, pulses_as_samples, samples_for_lags, pwr0, lagfr,
                                smsep, data_mask):
    """
    Ashton can write this.

    :param      num_range_gates:    The number range gates.
    :type       num_range_gates:    int
    :param      pulses_as_samples:  The pulses converted to sample number.
    :type       pulses_as_samples:  ndarray [num_records, num_pulses, num_ranges]
    :param      samples_for_lags:   The sample pairs used to create lags at each range.
    :type       samples_for_lags:   ndarray [num_records, num_lags+1, 2, num_ranges]
    :param      pwr0:               The lag 0 power.
    :type       pwr0:               ndarray [num_records, num_ranges]
    :param      lagfr:              The time to first range in us.
    :type       lagfr:              ndarray [num_records, ]
    :param      smsep:              The sample separation in us.
    :type       smsep:              ndarray [num_records, ]
    :param      data_mask:          The mask indicating where valid data from the file is.
    :type       data_mask:          ndarray [num_records, num_lags+1, 2, num_ranges]

    :returns:   Self clutter estimate at each range.
    :rtype:     ndarray [num_records, num_lags+1, num_ranges]
    """

    # [num_records, num_pulses, num_ranges]
    pulses_as_samples = data_fitting.Einsum.transpose(pulses_as_samples)[...,xp.newaxis,xp.newaxis,:,:]
    samples_for_lags = samples_for_lags[...,xp.newaxis]

    # [num_records, ]
    # [num_records, ]
    first_range_in_samps = lagfr/smsep

    # [num_records, num_lags+1, 2, num_ranges, 1]
    # [num_records, 1, 1, num_ranges, num_pulses]
    # [num_records, 1, 1, 1, 1]
    affected_ranges = (samples_for_lags -
                        pulses_as_samples -
                        first_range_in_samps[:,xp.newaxis,xp.newaxis,xp.newaxis,xp.newaxis])
    affected_ranges = affected_ranges.astype(int)

    ranges = xp.arange(num_range_gates)
    ranges_reshape = ranges[xp.newaxis,xp.newaxis,xp.newaxis,:,xp.newaxis]

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    # [num_records, num_lags+1, 2, num_ranges, 1]
    # [1, 1, 1, num_ranges, 1]
    # [num_records, num_lags+1, 2, num_ranges, 1]
    condition = ((affected_ranges <= samples_for_lags) &
                (affected_ranges < num_range_gates) &
                (affected_ranges != ranges_reshape) &
                (affected_ranges >= 0) &
                data_mask[...,xp.newaxis])

    # [num_records, 1, 1, num_ranges, 1]
    tmp_pwr = xp.array(xp.broadcast_to(pwr0[...,xp.newaxis,xp.newaxis,:,xp.newaxis], condition.shape))

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    tmp_pwr[~condition] = 0.0

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    affected_pwr = xp.sqrt(tmp_pwr)

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    # [num_records, 1, 1, num_ranges, 1]
    clutter_term12 = xp.einsum('...ijk,...kjl->...j', affected_pwr,
                                            xp.sqrt(pwr0[...,xp.newaxis,xp.newaxis,:,xp.newaxis]))

    affected_pwr = affected_pwr[...,xp.newaxis]

    pwr_1 = affected_pwr[:,:,0]
    pwr_2 = data_fitting.Einsum.transpose(affected_pwr[:,:,1])

    # [num_records, num_lags+1, num_ranges, num_pulses, 1]
    # [num_records, num_lags+1, num_ranges, 1, num_ranges]
    clutter_term3 = xp.einsum('...ijk,...ikm->...i', pwr_1, pwr_2)

    # [num_records, num_lags+1, num_ranges]
    # [num_records, num_lags+1, num_ranges]
    clutter = clutter_term12 + clutter_term3

    return clutter



def calculate_t(lags, mpinc):
    """
    Calculates the time  for each lag given the fundamental spacing in time. Also trim out the
    alternate lag 0.

    :param      lags:   The experiment lag table.
    :type       lags:   ndarray [num_records, num_lags+1, 2]
    :param      mpinc:  The fundamental lag time spacing.
    :type       mpinc:  ndarray [num_records, ]

    :returns:   The time spacing for each lag.
    :rtype:     ndarray [num_records, 1, num_lags]
    """

    # cut alt lag 0 and find difference between pulses.
    # [num_records, num_lags+1, 2]
    lags = lags[:,:-1,1] - lags[:,:-1,0]

    # Need braces to enforce data type. Lags is short and will overflow.
    # [num_records, num_lags]
    # [num_records, ]
    t = lags * (mpinc[...,xp.newaxis] * 1e-6)
    t = t[:, xp.newaxis,:]

    return t

def calculate_wavelength(tfreq):
    """
    Calculates the wavelengths for the frequencies used.

    :param      tfreq:  The frequencies(MHz)
    :type       tfreq:  ndarray [num_records, ]

    :returns:   Frequencies converted to wavelength.
    :rtype:     ndarray [num_records, 1, 1]
    """

    C = 299792458

    # [num_records, ]
    wavelength = C/(tfreq * 1e3)
    wavelength = wavelength[:,xp.newaxis,xp.newaxis]

    return wavelength

def calculate_constants(wavelength, t):
    """
    Pre-calculate the constants used in the model.

    :param      wavelength:  The wavelengths.
    :type       wavelength:  ndarray [num_records, 1, 1]
    :param      t:           The time spacing for each lag.
    :type       t:           ndarray [num_records, 1, num_lags]

    :returns:   The model constants.
    :rtype:     dict of ndarray [num_records, 1, 1, num_lags]
    """

    # [num_records, 1, num_lags]
    # [num_records, 1, 1]
    W_constant = (-1 * 2 * xp.pi * t)/wavelength
    W_constant = W_constant[...,xp.newaxis,:]

    # [num_records, 1, num_lags]
    # [num_records, 1, 1]
    V_constant = (1j * 4 * xp.pi * t)/wavelength
    V_constant = V_constant[...,xp.newaxis,:]

    return {'W' : W_constant, 'V' : V_constant}

def calculate_initial_W(array_shape):
    """
    Fills the initial spectral width guesses.

    :param      array_shape:  The array shape
    :type       array_shape:  ndarray [num_records, 1, num_models, 1]

    :returns:   The initial W guesses.
    :rtype:     ndarray [num_records, 1, num_models, 1]
    """

    W = xp.ones(array_shape) * 200.0

    return W

def calculate_initial_V(wavelength, num_velocity_models, mpinc):
    """
    Calculates the initial velocity guesses.

    :param      wavelength:           The wavelengths.
    :type       wavelength:           ndarray [num_records, 1, 1]
    :param      num_velocity_models:  The number of velocity models to fit.
    :type       num_velocity_models:  int
    :param      mpinc:                The fundamental lag time spacing.
    :type       mpinc:                ndarray [num_records, ]

    :returns:   The initial velocity guesses.
    :rtype:     ndarray [num_records, 1, num_models, 1]
    """

    # [num_records, 1, 1]
    # [num_records, 1, 1]
    nyquist_v = wavelength/(4.0 * mpinc[...,xp.newaxis,xp.newaxis] * 1e-6)

    # [num_records, 1, 1]
    step = 2 * nyquist_v / num_velocity_models

    tmp = xp.arange(num_velocity_models, dtype=xp.float64)[xp.newaxis,xp.newaxis,:]

    # [1, 1, num_models]
    # [num_records, 1, 1]
    tmp = tmp * step

    # [num_records, 1, 1]
    # [num_records, 1, num_models]
    nyquist_v_steps = (-1 * nyquist_v) + tmp

    nyquist_v_steps = nyquist_v_steps[...,xp.newaxis]

    return nyquist_v_steps


def compute_model_and_derivatives(params, **kwargs):
    """
    Calculates the model and derivatives used for fitting.

    :param      params:  The parameters that are being fit for.
    :type       params:  dict of fitting params.
    :param      kwargs:  The additional keywords arguments that will be supplied by the fitter.
    :type       kwargs:  dict that holds the model constants

    :returns:   The computed model and associated derivatives.
    :rtype:     dict of model and derivatives
    """

    p0 = params['p0']['values']
    W = params['W']['values']
    V = params['V']['values']

    model_constant_W = kwargs['W']
    model_constant_V = kwargs['V']

    # [num_records, num_ranges, num_models, 1]
    # [num_records, 1, 1, num_lags]
    # [num_records, num_ranges, num_models, 1]
    # [num_records, 1, 1, num_lags]
    # [num_records, num_ranges, num_models, 1]
    model = p0 * xp.exp(model_constant_W * W) * xp.exp(model_constant_V * V)

    J = xp.repeat(model[...,xp.newaxis], 3, axis=-1)

    # [num_records, num_ranges, num_models, num_lags]
    # [num_records, num_ranges, num_models, 1]
    J[:,:,:,:,0] /= p0

    # [num_records, num_ranges, num_models, num_lags]
    # [num_records, num_ranges, num_models, 1]
    J[:,:,:,:,1] *= model_constant_W

    # [num_records, num_ranges, num_models, num_lags]
    # [num_records, num_ranges, num_models, 1]
    J[:,:,:,:,2] *= model_constant_V

    model_dict = {}
    model = xp.concatenate((model.real, model.imag), axis=-1)

    # [num_records, num_ranges, num_models, num_lags*2]
    model_dict['model'] = model

    J = xp.concatenate((J.real, J.imag), axis=-2)

    # [num_records, num_ranges, num_models, num_lags*2, 3]
    model_dict['J'] = J

    return model_dict

def fit_all_records(records_data):
    """
    Top level function to initialize and perform fitting of SuperDARN rawacf data.

    :param      records_data:  Dictionary of data that has been reshaped into arrays.
    :type       records_data:  dict

    :returns:   Dictionary of fitted parameters
    :rtype:     dict
    """

    num_velocity_models = 30 # Number of steps in the Nyquist velocity space to fit.
    step = 10 # Number of records to fit at once.

    transmit_freq = xp.array(records_data['tfreq'])
    offset = xp.array(records_data['offset'])
    num_averages = xp.array(records_data['nave'])
    pwr0 = xp.array(records_data['pwr0'])
    acf = xp.array(records_data['acfd'])
    xcf = xp.array(records_data['xcfd'])
    lags = xp.array(records_data['ltab'])
    pulses = xp.array(records_data['ptab'])

    mpinc = xp.array(records_data['mpinc'])
    lagfr = xp.array(records_data['lagfr'])
    smsep = xp.array(records_data['smsep'])
    num_range_gates = acf.shape[1]

    # This mask indicates where input data is valid
    data_mask = xp.array(records_data["data_mask"])

    # Transpose and reshape into form X = [R I]
    acf = data_fitting.Einsum.transpose(acf)
    acf = acf.reshape((acf.shape[0], acf.shape[1], acf.shape[2] * acf.shape[3]))

    t = calculate_t(lags, mpinc)
    wavelength = calculate_wavelength(transmit_freq)

    noise = determine_noise(pwr0)

    pulses_as_samples, samples_for_lags = calculate_samples(num_range_gates, lags, pulses,
                                                                mpinc, lagfr, smsep)
    blanking_mask = create_blanking_mask(num_range_gates, pulses_as_samples, samples_for_lags,
                                            data_mask, num_averages)

    good_points = xp.count_nonzero(blanking_mask == False, axis=-1)
    good_points = good_points[...,xp.newaxis,xp.newaxis]

    clutter = estimate_max_self_clutter(num_range_gates, pulses_as_samples, samples_for_lags,
                                        pwr0, lagfr, smsep, data_mask)
    fo_weights = first_order_weights(pwr0, noise, clutter, num_averages, blanking_mask)

    model_constants = calculate_constants(wavelength, t)

    initial_V = calculate_initial_V(wavelength, num_velocity_models, mpinc)
    initial_W = calculate_initial_W(initial_V.shape)
    initial_p0 = pwr0[...,xp.newaxis,xp.newaxis]

    total_records = pwr0.shape[0]
    consistent_shape = (total_records, num_range_gates, num_velocity_models, 1)

    initial_V = xp.broadcast_to(initial_V, consistent_shape).copy()
    initial_W = xp.broadcast_to(initial_W, consistent_shape).copy()
    initial_p0 = xp.broadcast_to(initial_p0, consistent_shape).copy()

    even_chunks = total_records // step
    remainder = total_records - (even_chunks * step)

    fits = []
    def do_fits(args):
        """
        Helper function to perform fitting on a chunk of records.
        """

        start = args[0]
        stop = args[1]

        W_i = initial_W[start:stop,...]
        V_i = initial_V[start:stop,...]
        p0_i = initial_p0[start:stop,...]

        V_upper_bound = xp.repeat(initial_V[start:stop,:,-1,xp.newaxis], num_velocity_models, axis=2)
        V_lower_bound = xp.repeat(initial_V[start:stop,:,0,xp.newaxis], num_velocity_models, axis=2)

        params = {'p0': {'values' : p0_i, 'min' : 0.0001, 'max' : None},
              'W': {'values' : W_i, 'min' : -200.0, 'max' : 8000.0},
              'V': {'values' : V_i, 'min' : V_lower_bound, 'max' : V_upper_bound}}


        W_constant_i = model_constants['W'][start:stop,...]
        V_constant_i = model_constants['V'][start:stop,...]

        model_constants_i = {'W' : W_constant_i,
                             'V' : V_constant_i}
        model_dict = {'model_fn' : compute_model_and_derivatives, 'args' : model_constants_i}

        weights = fo_weights[start:stop,...]

        acf_i = acf[start:stop,...]

        lm_step_shape = (stop-start, num_range_gates, num_velocity_models, 1)

        gp = good_points[start:stop,...]

        return data_fitting.LMFit(acf_i, model_dict, weights, params, lm_step_shape, gp)

    argv = []
    if even_chunks > 0:
        for i in range(even_chunks):
            argv.append((i*step, (i+1)*step))

    if remainder:
        argv.append((even_chunks*step, total_records))

    p = ThreadPool()

    fits = p.map(do_fits, argv)

    tmp = ([], [], [], [], [], [])

    for f in fits:
        tmp[0].append(f.fitted_params['p0']['values'])
        tmp[1].append(f.fitted_params['W']['values'])
        tmp[2].append(f.fitted_params['V']['values'])
        tmp[3].append(f.cov_mat)
        tmp[4].append(f.converged)
        tmp[5].append(f.chi_2)

    fitted_data = {'p0' : xp.vstack(tmp[0]).asnumpy(),
                   'W' : xp.vstack(tmp[1]).asnumpy(),
                   'V' : xp.vstack(tmp[2]).asnumpy(),
                   'cov_mat' : xp.vstack(tmp[3]).asnumpy(),
                   'converged' : xp.vstack(tmp[4]).asnumpy(),
                   'chi_2': xp.vstack(tmp[5]).asnumpy()}


    return fitted_data

def write_to_file(records_data, fitted_data, output_name):
    """
    Writes fitting output to hdf5 output file.

    :param      records_data:  The file records data
    :type       records_data:  dict
    :param      fitted_data:   The fitted parameters and statistics.
    :type       fitted_data:   dict
    :param      output_name:   The output file name
    :type       output_name:   str
    """

    output_data = {}

    records_data.pop('acfd', None)
    records_data.pop('xcfd', None)
    records_data.pop('slist', None)
    records_data.pop('pwr0', None)
    records_data.pop('data_mask', None)

    for k,v in records_data.items():
        output_data[k] = v

    for k,v in fitted_data.items():
        output_data[k] = v

    dd.io.save(output_name, output_data, compression='zlib')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fit SuperDARN data using Reimer-Hussey 2018 method')
    parser.add_argument('--use-gpu', action='store_true', help='Use gpu if available')
    parser.add_argument('filename', type=str, help='Input file path to process')

    args = parser.parse_args()


    input_file = args.filename

    data_read = rdr.RawacfDmapRead(input_file)

    records_data = data_read.get_parsed_data()


    if args.use_gpu:
        try:
            import cupy as xp
        except ImportError:
            print('Cupy not available, falling back to Numpy')
            import numpy as xp
    else:
        import numpy as xp

    fitted_data = fit_all_records(records_data)

    output_name = input_file.split('.')
    output_name[-1] = 'rh18.hdf5'
    output_name = ".".join(output_name)

    write_to_file(records_data, fitted_data, output_name)

















