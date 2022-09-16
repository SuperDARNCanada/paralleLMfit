import rawacf_dmap_read as rdr
import data_fitting
import sys
import time
import argparse
import deepdish as dd
import copy
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from backscatter import dmap

V_MAX = 30.0
W_MAX = 90.0


def sdarn_determinations(fit_dict, noise, blanking_mask):
    pwr0 = fit_dict['p0']
    wid = fit_dict['W']
    vel = fit_dict['V']
    pwr0_err = fit_dict['p0_err']
    wid_err = fit_dict['W_err']
    vel_err = fit_dict['V_err']
    chi_2 = fit_dict['chi_2']

    num_records = pwr0.shape[0]

    noise = noise[..., xp.newaxis]

    sdarn_dict = {}
    sdarn_dict['fitacf.revision.major'] = [xp.int32(3)] * num_records
    sdarn_dict['fitacf.revision.minor'] = [xp.int32(0)] * num_records
    sdarn_dict['noise.sky'] = noise.reshape(noise.size)
    sdarn_dict['noise.lag0'] = xp.zeros(num_records)
    sdarn_dict['noise.vel'] = xp.zeros(num_records)

    sdarn_dict['nlag'] = xp.count_nonzero(~blanking_mask, axis=-1) // 2     # TODO: Figure out why this differs

    p0_db = 10.0 * xp.log10((pwr0 - noise) / noise)
    p0_db[pwr0 - noise <= 1e-5] = -50.0
    sdarn_dict['pwr0'] = xp.array(p0_db)            # TODO: Figure out how much this differs

    sdarn_dict['qflg'] = xp.ones(pwr0.shape, dtype=xp.int8)

    # TODO: Refractive index correction?
    # sdarn_dict['refrc_idx'] = ?
    # velocity = vel * (1 / refractive_index)
    # velocity_err = xp.abs(vel_err) * (1 / refractive_index)
    sdarn_dict['p_l'] = 10 * xp.log10(pwr0 / noise)
    sdarn_dict['p_l_e'] = xp.sqrt(pwr0_err) / (xp.log(10.0) * pwr0)
    sdarn_dict['v'] = xp.array(vel)                         # TODO: Figure out how much this differs
    sdarn_dict['v_e'] = xp.sqrt(vel_err)                    # TODO: Figure out how much this differs
    sdarn_dict['w_l'] = xp.array(wid)                       # TODO: Figure out how much this differs
    sdarn_dict['w_l_e'] = xp.sqrt(wid_err)                  # TODO: Figure out how much this differs
    sdarn_dict['sd_l'] = xp.array(chi_2)

    groundscatter = xp.zeros(vel.shape, dtype=xp.int8)
    groundscatter[xp.abs(vel) - wid * (V_MAX/W_MAX) < 0] = 1
    sdarn_dict['gflg'] = groundscatter

    # TODO: Quadratic fit
    sdarn_dict['p_s'] = xp.ones(pwr0.shape) * -xp.inf
    sdarn_dict['p_s_e'] = xp.ones(pwr0.shape) * xp.nan
    sdarn_dict['w_s'] = xp.zeros(wid.shape)
    sdarn_dict['w_s_e'] = xp.zeros(wid_err.shape)
    sdarn_dict['sd_s'] = xp.zeros(chi_2.shape)

    # TODO: Elevation fitting
    sdarn_dict['xcf'] = xp.ones(num_records, dtype=xp.int16)     # Flag on whether xcf included for each record
    sdarn_dict['x_qflg'] = xp.zeros(pwr0.shape, dtype=xp.int8)
    sdarn_dict['x_gflg'] = xp.zeros(groundscatter.shape, dtype=xp.int8)
    sdarn_dict['phi0'] = xp.zeros(vel.shape)
    sdarn_dict['phi0_e'] = xp.zeros(vel.shape)
    sdarn_dict['sd_phi'] = xp.ones(vel.shape) * -1
    sdarn_dict['x_p_l'] = xp.zeros(pwr0.shape)
    sdarn_dict['x_p_l_e'] = xp.zeros(pwr0.shape)
    sdarn_dict['x_p_s'] = xp.zeros(pwr0.shape)
    sdarn_dict['x_p_s_e'] = xp.zeros(pwr0.shape)
    sdarn_dict['x_v'] = xp.zeros(vel.shape)
    sdarn_dict['x_v_e'] = xp.zeros(vel.shape)
    sdarn_dict['x_w_l'] = xp.zeros(wid.shape)
    sdarn_dict['x_w_l_e'] = xp.zeros(wid.shape)
    sdarn_dict['x_w_s'] = xp.zeros(wid.shape)
    sdarn_dict['x_w_s_e'] = xp.zeros(wid.shape)
    sdarn_dict['x_sd_l'] = xp.zeros(chi_2.shape)
    sdarn_dict['x_sd_s'] = xp.zeros(chi_2.shape)
    sdarn_dict['x_sd_phi'] = xp.zeros(chi_2.shape)
    sdarn_dict['elv'] = xp.ones(vel.shape) * -1
    sdarn_dict['elv_low'] = xp.zeros(vel.shape)
    sdarn_dict['elv_high'] = xp.ones(vel.shape) * -1

    return sdarn_dict


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
    noise = xp.mean(sorted_pwr0[:, 0:10], axis=1)
    return noise


def estimate_re_im_error(t, pwr0, width, vel, noise, clutter, nave, blanking_mask, wavelength):
    """
    Use an exponential decay model for the ACF to evaluate the error of the real and imaginary components
    of fitted ACFs.

    :param      t:              The time spacing for each lag.
    :type       t:              ndarray [num_records, 1, num_lags]
    :param      pwr0:           The lag0 power.
    :type       pwr0:           ndarray [num_records, num_ranges, 1]
    :param      width:          The fitted spectral width.
    :type       width:          ndarray [num_records, num_ranges, 1]
    :param      vel:            The fitted velocity.
    :type       vel:            ndarray [num_records, num_ranges, 1]
    :param      noise:          The noise estimates.
    :type       noise:          ndarray [num_records, ]
    :param      clutter:        The clutter estimates.
    :type       clutter:        ndarray [num_records, num_lags+1, num_ranges]
    :param      nave:           The number of averages.
    :type       nave:           ndarray [num_records, ]
    :param      blanking_mask:  The blanking mask to indicate bad points.
    :type       blanking_mask:  ndarray [num_records, num_ranges, num_lags*2]
    :param      wavelength:     The wavelength of the transmitted signal.
    :type       wavelength:     ndarray [num_records, 1, 1]

    :returns:   The estimated real and imaginary fitted errors.
    :rtype:     ndarray [num_records, num_ranges, num_lags*2]
    """

    clutter = data_fitting.Einsum.transpose(clutter)[..., :-1]

    # [num_records, num_ranges, 1]
    # [num_records, 1, 1]
    # [num_records, num_ranges, num_lags]
    received_power = (pwr0 + noise[:, xp.newaxis, xp.newaxis] + clutter)

    # [num_records, num_ranges, 1]
    # [num_records, 1, num_lags]
    # [num_records, 1, 1]
    rho = xp.exp(-2 * xp.pi * width * t / wavelength)
    rho[rho > 0.999] = 0.999

    # [num_records, num_ranges, num_lags]
    # [num_records, num_ranges, 1]
    # [num_records, num_ranges, num_lags]
    rho = rho * pwr0 / received_power

    # [num_records, num_ranges, num_lags]
    # [num_records, num_ranges, 1]
    # [num_records, 1, num_lags]
    # [num_records, 1, 1]
    rho_real = rho * xp.cos(4 * xp.pi * vel * t / wavelength)
    rho_imag = rho * xp.sin(4 * xp.pi * vel * t / wavelength)

    # [num_records, num_ranges, num_lags]
    # [num_records, num_ranges, num_lags]
    # [num_records, 1, 1]
    real_error = received_power * xp.sqrt((1 - rho*rho)/2.0 + (rho_real*rho_real)) / \
                 xp.sqrt(nave[..., xp.newaxis, xp.newaxis].astype(float))
    imag_error = received_power * xp.sqrt((1 - rho*rho)/2.0 + (rho_imag*rho_imag)) / \
                 xp.sqrt(nave[..., xp.newaxis, xp.newaxis].astype(float))

    # [num_records, num_ranges, num_lags*2]
    weights_shape = (real_error.shape[0], real_error.shape[1], real_error.shape[2] * 2)
    weights = xp.zeros(weights_shape)

    weights[..., 0:real_error.shape[2]] = 1 / (real_error ** 2)
    weights[..., real_error.shape[2]:] = 1 / (imag_error ** 2)

    weights[blanking_mask] = 1e-20

    return weights


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
    :rtype:     ndarray [num_records, num_ranges, num_lags*2]
    """

    # [num_records, 1, num_ranges]
    # [num_records, 1, 1]
    # [num_records, num_lags+1, num_ranges]
    received_power = (pwr0[:, xp.newaxis, :] + noise[:, xp.newaxis, xp.newaxis] + clutter)

    # [num_records, num_lags+1, num_ranges]
    # [num_records, 1, 1]
    error = received_power / xp.sqrt(nave[:, xp.newaxis, xp.newaxis].astype(float))
    error = data_fitting.Einsum.transpose(error)

    # Cut alt lag 0
    # [num_records, num_ranges, num_lags+1]
    error = error[..., :-1]

    weights_shape = (error.shape[0], error.shape[1], error.shape[2] * 2)
    # [num_records, num_ranges, num_lags*2]
    weights = xp.zeros(weights_shape)

    weights[..., 0:error.shape[2]] = 1.0 / (error ** 2)
    weights[..., error.shape[2]:] = 1.0 / (error ** 2)

    # [num_records, num_ranges, num_lags*2]
    weights[..., :][blanking_mask] = 1e-20

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
    pulses_by_ranges = xp.repeat(ptab[..., xp.newaxis], num_range_gates, axis=-1)

    # [num_records, ]
    # [num_records, ]
    ranges_per_tau = mpinc / smsep

    ranges_per_tau = ranges_per_tau[..., xp.newaxis, xp.newaxis]

    # [num_records, num_pulses, num_ranges]
    # [num_records, 1, 1]
    pulses_as_samples = (pulses_by_ranges * ranges_per_tau)

    # [num_records, ]
    # [num_records, ]
    first_range_in_samps = lagfr / smsep

    # [num_ranges, ]
    # [num_records, 1, 1, 1]
    range_off = ranges + first_range_in_samps[..., xp.newaxis, xp.newaxis, xp.newaxis]

    # [num_records, num_lags+1, 2]
    # [num_records, ]
    lags_pairs_as_samples = ltab * ranges_per_tau

    # [num_records, 1, 1, num_ranges]
    # [num_records, num_lags+1, 2, 1]
    # TODO: Take into account alternate lag0 swapping
    samples_for_lags = range_off + lags_pairs_as_samples[..., xp.newaxis]

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
    blanked_1 = data_fitting.Einsum.transpose(pulses_as_samples)[..., xp.newaxis, xp.newaxis, :, :]
    blanked_2 = blanked_1 + 1

    # [num_records, 1, 1, num_ranges, num_pulses]
    # [num_records, num_lags+1, 2, num_ranges, 1]
    x = blanked_1 == samples_for_lags[..., xp.newaxis]
    y = blanked_2 == samples_for_lags[..., xp.newaxis]

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    blanking_mask = xp.any(x | y, axis=-1)

    # Blank where data does not exist
    # [num_records, num_lags+1, 2, num_ranges]
    blanking_mask[~data_mask] = True

    # If either sample which contributes to a lag is blanked, then blank the whole lag
    lag_mask = xp.any(blanking_mask, axis=2, keepdims=True)
    lag_mask[:, 0, ...] = False     # lag0 has no blanks - alternate lag0 used to fill them in
    blanking_mask = xp.repeat(lag_mask, 2, axis=2)

    # [num_records, ]
    blanking_mask[num_averages <= 0] = True

    # [num_records, num_lags+1, 2, num_ranges]
    blanking_mask = xp.einsum('...ijk->...kji', blanking_mask)

    # cut alt lag 0
    # [num_records, num_ranges, 2, num_lags+1]
    blanking_mask = blanking_mask[..., :-1]

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
    pulses_as_samples = data_fitting.Einsum.transpose(pulses_as_samples)[..., xp.newaxis, xp.newaxis, :, :]
    samples_for_lags = samples_for_lags[..., xp.newaxis]

    # [num_records, ]
    # [num_records, ]
    first_range_in_samps = lagfr / smsep

    # [num_records, num_lags+1, 2, num_ranges, 1]
    # [num_records, 1, 1, num_ranges, num_pulses]
    # [num_records, 1, 1, 1, 1]
    affected_ranges = (samples_for_lags -
                       pulses_as_samples -
                       first_range_in_samps[:, xp.newaxis, xp.newaxis, xp.newaxis, xp.newaxis])
    affected_ranges = affected_ranges.astype(int)

    ranges = xp.arange(num_range_gates)
    ranges_reshape = ranges[xp.newaxis, xp.newaxis, xp.newaxis, :, xp.newaxis]

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    # [num_records, num_lags+1, 2, num_ranges, 1]
    # [1, 1, 1, num_ranges, 1]
    # [num_records, num_lags+1, 2, num_ranges, 1]
    condition = ((affected_ranges < num_range_gates) &
                 (affected_ranges != ranges_reshape) &
                 (affected_ranges >= 0) &
                 data_mask[..., xp.newaxis])

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    affected_ranges[~condition] = 0

    # [num_records, num_ranges]
    # [num_records, -1]
    tmp_pwr = xp.take_along_axis(pwr0, affected_ranges.reshape(affected_ranges.shape[0], -1), axis=1)
    tmp_pwr = tmp_pwr.reshape(affected_ranges.shape)
    tmp_pwr[~condition] = 0

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    affected_pwr = xp.sqrt(tmp_pwr)

    # [num_records, num_lags+1, 2, num_ranges, num_pulses]
    # [num_records, 1, 1, num_ranges, 1]
    clutter_term12 = xp.einsum('...ijk,...kjl->...j', affected_pwr,
                               xp.sqrt(pwr0[..., xp.newaxis, xp.newaxis, :, xp.newaxis]))

    affected_pwr = affected_pwr[..., xp.newaxis]

    pwr_1 = affected_pwr[:, :, 0]
    pwr_2 = data_fitting.Einsum.transpose(affected_pwr[:, :, 1])

    # [num_records, num_lags+1, num_ranges, num_pulses, 1]
    # [num_records, num_lags+1, num_ranges, 1, num_pulses]
    clutter_term3 = xp.einsum('...ijk,...ikm->...i', pwr_1, pwr_2)

    # [num_records, num_lags+1, num_ranges]
    # [num_records, num_lags+1, num_ranges]
    clutter = clutter_term12 + clutter_term3

    return clutter


def calculate_t(lags, mpinc):
    """
    Calculates the time for each lag given the fundamental spacing in time. Also trim out the
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
    lags = lags[:, :-1, 1] - lags[:, :-1, 0]

    # Need braces to enforce data type. Lags is short and will overflow.
    # [num_records, num_lags]
    # [num_records, ]
    t = lags * (mpinc[..., xp.newaxis] * 1e-6)
    t = t[:, xp.newaxis, :]

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
    wavelength = C / (tfreq * 1e3)
    wavelength = wavelength[:, xp.newaxis, xp.newaxis]

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
    W_constant = (-1 * 2 * xp.pi * t) / wavelength
    W_constant = W_constant[..., xp.newaxis, :]

    # [num_records, 1, num_lags]
    # [num_records, 1, 1]
    V_constant = (1j * 4 * xp.pi * t) / wavelength
    V_constant = V_constant[..., xp.newaxis, :]

    return {'W': W_constant, 'V': V_constant}


def calculate_initial_W(array_shape):
    """
    Fills the initial spectral width guesses.

    :param      array_shape:  The array shape
    :type       array_shape:  ndarray [num_records, 1, num_models]

    :returns:   The initial W guesses.
    :rtype:     ndarray [num_records, 1, num_models]
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
    nyquist_v = wavelength / (4.0 * mpinc[..., xp.newaxis, xp.newaxis] * 1e-6)

    # [num_records, 1, 1]
    step = 2 * nyquist_v / num_velocity_models

    tmp = xp.arange(num_velocity_models, dtype=xp.float64)[xp.newaxis, xp.newaxis, :] + 0.5

    # [1, 1, num_models]
    # [num_records, 1, 1]
    tmp = tmp * step

    # [num_records, 1, 1]
    # [num_records, 1, num_models]
    nyquist_v_steps = (-1 * nyquist_v) + tmp

    return nyquist_v_steps


def compute_model_and_derivatives(x_data, params, **kwargs):
    """
    Calculates the model and derivatives used for fitting.

    :param      x_data:  The independent variable values (unused)
    :type       x_data:  ndarray [..., m_points]
    :param      params:  The parameters that are being fit for.
    :type       params:  ndarray [..., n_params, 1]
    :param      kwargs:  The additional keywords arguments that will be supplied by the fitter.
    :type       kwargs:  dict that holds the model constants

    :returns:   The computed model and associated derivatives.
    :rtype:     tuple of model and derivatives
    """

    p0 = params[..., [0], :]
    W = params[..., [1], :]
    V = params[..., [2], :]

    model_constant_W = kwargs['W']
    model_constant_V = kwargs['V']

    # [num_records, num_ranges, num_models, 1]
    # [num_records, 1, 1, num_lags]
    # [num_records, num_ranges, num_models, 1]
    # [num_records, 1, 1, num_lags]
    # [num_records, num_ranges, num_models, 1]
    model = p0 * xp.exp(model_constant_W * W) * xp.exp(model_constant_V * V)

    J = xp.repeat(model, 3, axis=-1)

    # [num_records, num_ranges, num_models, num_lags]
    # [num_records, num_ranges, num_models, 1]
    J[..., [0]] /= p0

    # [num_records, num_ranges, num_models, num_lags]
    # [num_records, num_ranges, num_models, 1]
    J[..., [1]] *= model_constant_W

    # [num_records, num_ranges, num_models, num_lags]
    # [num_records, num_ranges, num_models, 1]
    J[..., [2]] *= model_constant_V

    model = xp.concatenate((model.real, model.imag), axis=-2)
    J = xp.concatenate((J.real, J.imag), axis=-2)

    return model, J


def fitted_params_and_errors(fit, confidence):
    """
    Extracts the relevant parameters from a fit with a given confidence (number of standard deviations),
    assuming Gaussian fitted parameter errors.

    :param         fit: Fitted LMFit object
    :type          fit: LMFit
    :param  confidence: Confidence interval in terms of standard deviations
    :type   confidence: float
    """
    if confidence <= 0:
        raise ValueError("Confidence interval must be a positive number.")
    delta_chi2 = confidence * confidence

    num_params = fit.fitted_params.shape[-1]

    # # Get the parameters and errors for the best fit of the bunch
    best_fit_idx = xp.argmin(fit.chi_2, axis=-1, keepdims=True)
    best_fitted_params = xp.take_along_axis(fit.fitted_params, best_fit_idx[..., xp.newaxis], axis=-2)

    fitted_param_errors = xp.einsum('...ii->...i', fit.cov_mat) * delta_chi2
    best_fitted_param_errors = xp.take_along_axis(fitted_param_errors, best_fit_idx[..., xp.newaxis], axis=-2)

    # If there are local minima within delta_chi2 significance of the global minimum, then update the error accordingly
    global_min_chi2 = xp.take_along_axis(fit.chi_2, best_fit_idx, axis=-1)
    local_minima_mask = fit.chi_2 <= global_min_chi2 + delta_chi2
    local_minima_mask = xp.repeat(local_minima_mask[..., xp.newaxis], num_params, axis=-1)
    param_deviations = xp.ma.array(xp.abs(fit.fitted_params - best_fitted_params), mask=~local_minima_mask)
    local_minima_param_errors = param_deviations.max(axis=-2)
    best_fitted_param_errors = xp.array(xp.maximum(best_fitted_param_errors[..., 0, :], local_minima_param_errors))

    fit_dict = {'chi_2': global_min_chi2[..., 0],
                'params': best_fitted_params[..., 0, :],
                'errors': best_fitted_param_errors}
    return fit_dict


def fit_all_records(records_data):
    """
    Top level function to initialize and perform fitting of SuperDARN rawacf data.

    :param      records_data:  Dictionary of data that has been reshaped into arrays.
    :type       records_data:  dict

    :returns:   Dictionary of fitted parameters
    :rtype:     dict
    """

    num_velocity_models = 30  # Number of steps in the Nyquist velocity space to fit.
    step = 10  # Number of records to fit at once.

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

    pulses_as_samples, samples_for_lags = calculate_samples(num_range_gates, lags, pulses, mpinc, lagfr, smsep)
    blanking_mask = create_blanking_mask(num_range_gates, pulses_as_samples, samples_for_lags, data_mask, num_averages)

    good_points = xp.count_nonzero(blanking_mask == False, axis=-1)
    good_points = good_points[..., xp.newaxis, xp.newaxis]

    clutter = estimate_max_self_clutter(num_range_gates, pulses_as_samples, samples_for_lags, pwr0, lagfr, smsep,
                                        data_mask)
    fo_weights = first_order_weights(pwr0, noise, clutter, num_averages, blanking_mask)

    model_constants = calculate_constants(wavelength, t)

    initial_V = calculate_initial_V(wavelength, num_velocity_models, mpinc)
    initial_W = calculate_initial_W(initial_V.shape)
    initial_p0 = pwr0[..., xp.newaxis]

    total_records = pwr0.shape[0]
    consistent_shape = (total_records, num_range_gates, num_velocity_models)

    initial_V = xp.broadcast_to(initial_V, consistent_shape).copy()
    initial_W = xp.broadcast_to(initial_W, consistent_shape).copy()
    initial_p0 = xp.broadcast_to(initial_p0, consistent_shape).copy()

    even_chunks = total_records // step
    remainder = total_records - (even_chunks * step)

    def do_fits(args):
        """
        Helper function to perform fitting on a chunk of records.
        """

        start = args[0]
        stop = args[1]

        # [num_records, num_ranges, num_models]
        p0_i = initial_p0[start:stop, ...]
        W_i = initial_W[start:stop, ...]
        V_i = initial_V[start:stop, ...]

        # [num_records, num_ranges, num_models]
        params = xp.stack([p0_i, W_i, V_i], axis=-1)

        # hardcoded lower and upper bounds for pwr0, spectral width, and LOS velocity
        bounds = xp.array([[0.0001, -200.0, initial_V[0, 0, 0]], [xp.inf, 8000.0, initial_V[0, 0, -1]]])
        bounds = xp.transpose(bounds)

        W_constant_i = model_constants['W'][start:stop, ..., xp.newaxis]
        V_constant_i = model_constants['V'][start:stop, ..., xp.newaxis]

        weights = fo_weights[start:stop, ...]

        acf_i = acf[start:stop, ...]

        num_records = stop - start
        num_ranges = acf_i.shape[1]
        n_points = t.shape[-1]

        x_data = xp.repeat(t[start:stop, ...], 2, axis=-1)
        x_data = xp.reshape(x_data, (num_records, 1, 1, n_points*2))
        y_data = xp.broadcast_to(acf_i[..., xp.newaxis, :], (num_records, num_ranges, 1, n_points*2))
        weights = weights[..., xp.newaxis, :]
        num_points = good_points[start:stop, ...]

        fit1 = data_fitting.LMFit(compute_model_and_derivatives, x_data, y_data, params, weights,
                                  bounds=bounds, num_points=num_points, W=W_constant_i, V=V_constant_i)

        # Get the parameters and errors for the best fit of the bunch
        best_fit_idx = xp.argmin(fit1.chi_2, axis=-1, keepdims=True)
        fitted_params = xp.take_along_axis(fit1.fitted_params, best_fit_idx[..., xp.newaxis], axis=-2)

        weights = estimate_re_im_error(t[start:stop, ...], fitted_params[..., 0], fitted_params[..., 1],
                                       fitted_params[..., 2], noise, clutter, num_averages, blanking_mask, wavelength)
        weights = weights[..., xp.newaxis, :]

        fit2 = data_fitting.LMFit(compute_model_and_derivatives, x_data, y_data, params, weights,
                                  bounds=bounds, num_points=num_points, W=W_constant_i, V=V_constant_i)

        # TODO: Fit the XCF

        confidence = 2  # TODO: Make this configurable
        fit_dict = fitted_params_and_errors(fit2, confidence)

        return fit_dict

    argv = []
    if even_chunks > 0:
        for i in range(even_chunks):
            argv.append((i * step, (i + 1) * step))

    if remainder:
        argv.append((even_chunks * step, total_records))

    p = ThreadPool()

    fits = p.map(do_fits, argv)

    tmp = ([], [], [], [], [], [], [])

    for f in fits:

        tmp[0].append(f['params'][..., 0])
        tmp[1].append(f['params'][..., 1])
        tmp[2].append(f['params'][..., 2])
        tmp[3].append(f['errors'][..., 0])
        tmp[4].append(f['errors'][..., 1])
        tmp[5].append(f['errors'][..., 2])
        tmp[6].append(f['chi_2'])

    fitted_data = {'p0': xp.vstack(tmp[0]),
                   'W': xp.vstack(tmp[1]),
                   'V': xp.vstack(tmp[2]),
                   'p0_err': xp.vstack(tmp[3]),
                   'W_err': xp.vstack(tmp[4]),
                   'V_err': xp.vstack(tmp[5]),
                   'chi_2': xp.vstack(tmp[6])}
    for k, v in fitted_data.items():
        print(f'{k}: {v.shape}')

    fit_file_params = sdarn_determinations(fitted_data, noise, blanking_mask)

    # return fitted_data, t, wavelength, acf, fo_weights
    return fit_file_params


def dict_to_record_list(param_dict):
    """Converts a dictionary of params for each record into a list of dictionaries, one per record."""
    record_dicts = []
    num_records = param_dict['cp'].shape[0]    # Chose cp arbitrarily, any field should work.

    for record in range(num_records):
        record_dict = {}
        for k, v in param_dict.items():
            record_dict[k] = v[record]
        record_dicts.append(record_dict)

    return record_dicts


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
    # records_data.pop('slist', None)
    records_data.pop('mplgexs', None)
    records_data.pop('pwr0', None)
    records_data.pop('data_mask', None)
    records_data.pop('ifmode', None)
    records_data.pop('rawacf.revision.major', None)
    records_data.pop('rawacf.revision.minor', None)
    records_data.pop('thr')

    for k, v in records_data.items():
        output_data[k] = v

    for k, v in fitted_data.items():
        output_data[k] = v

    record_dicts = dict_to_record_list(output_data)
    dmap.dicts_to_file(record_dicts, output_name, file_type='fitacf')
    # dd.io.save(output_name, output_data, compression='zlib')


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

    # fitted_data, t, wavelength, acf, weights = fit_all_records(records_data)
    fitted_data = fit_all_records(records_data)

    # # This section is for plotting - for testing purposes
    # p0 = fitted_data['p0']
    # W = fitted_data['W']
    # V = fitted_data['V']
    #
    # for i in range(27, 36):
    #     pwr = p0[0, i]
    #     wid = W[0, i]
    #     vel = V[0, i]
    #
    #     print('pwr, w, v: ', pwr, wid, vel)
    #
    #     model = pwr * xp.exp(-2 * xp.pi * t[0, 0] * wid / wavelength[0, 0, 0]) * \
    #                   xp.exp(1j * 4 * xp.pi * vel * t[0, 0] / wavelength[0, 0, 0])
    #     re_model = model.real
    #     im_model = model.imag
    #
    #     blanks = weights < 1e-19
    #     weights[blanks] = xp.nan
    #     acf[blanks] = xp.nan
    #     error = 1 / xp.sqrt(weights)
    #     num_lags = weights.shape[-1] // 2
    #
    #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
    #     ax1.errorbar(t[0, 0], acf[0, i, :num_lags], yerr=error[0, i, :num_lags], marker='o', color='blue', label='Real')
    #     ax2.errorbar(t[0, 0], acf[0, i, num_lags:], yerr=error[0, i, num_lags:], marker='o', color='blue', label='Imag')
    #     ax1.plot(t[0, 0], re_model, color='red', label='Fitted Real')
    #     ax2.plot(t[0, 0], im_model, color='red', label='Fitted Imag')
    #     ax1.legend()
    #     ax2.legend()
    #     ax2.set_xlabel('Time (s)')
    #     ax1.set_ylabel('Amplitude')
    #     ax2.set_ylabel('Amplitude')
    #     plt.show()
    #     plt.close()

    output_name = input_file.split('.')
    # output_name[-1] = 'rh18.hdf5'
    output_name[-1] = 'lmfit'
    output_name = ".".join(output_name)

    write_to_file(records_data, fitted_data, output_name)
