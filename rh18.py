import rawacf_dmap_read as rdr
import data_fitting
import sys
import time
import argparse
import deepdish as dd

from multiprocessing.pool import ThreadPool

def determine_noise(pwr0):

    sorted_pwr0 = xp.sort(pwr0)
    noise = xp.mean(sorted_pwr0[:,0:10], axis=1)
    return noise

def first_order_weights(pwr0, noise, clutter, nave, blanking_mask):

    total_signal = (pwr0[:,xp.newaxis,:] + noise[:,xp.newaxis,xp.newaxis] + clutter)

    error = total_signal / xp.sqrt(nave[:,xp.newaxis,xp.newaxis].astype(float))
    error = data_fitting.Einsum.transpose(error)

    # Cut alt lag 0
    error = error[...,:-1]

    weights_shape = (error.shape[0], error.shape[1], error.shape[2] * 2, error.shape[2] * 2)
    weights = xp.zeros(weights_shape)

    diag = xp.arange(weights.shape[-1])
    diag_r = diag[0:error.shape[2]]
    diag_i = diag[error.shape[2]:]

    weights[...,diag_r,diag_r] = error
    weights[...,diag_i,diag_i] = error

    weights[...,diag,diag][blanking_mask] = 1e20

    return weights


def calculate_samples(num_range_gates, ltab, ptab, mpinc, lagfr, smsep):

    ranges = xp.arange(num_range_gates)
    pulses_by_ranges = xp.repeat(ptab[...,xp.newaxis], num_range_gates, axis=-1)

    ranges_per_tau = mpinc / smsep
    ranges_per_tau = ranges_per_tau[...,xp.newaxis,xp.newaxis]

    pulses_as_samples = (pulses_by_ranges * ranges_per_tau)
    first_range_in_samps = lagfr/smsep
    first_range_in_samps = first_range_in_samps[...,xp.newaxis,xp.newaxis]
    samples = (pulses_by_ranges * ranges_per_tau) + ranges[xp.newaxis,xp.newaxis,:] + first_range_in_samps

    def searchsorted2d(a,b):
        """https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy"""
        m,n,o = a.shape
        max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
        r = max_num*xp.arange(a.shape[0])[:,xp.newaxis,xp.newaxis]
        p = xp.searchsorted( (a+r).ravel(), (b+r).ravel() ).reshape(b.shape)
        return p - n*(xp.arange(m)[:,xp.newaxis, xp.newaxis])

    lags_pulse_idx = searchsorted2d(ptab[...,xp.newaxis], ltab)
    lags_pulse_idx = xp.repeat(lags_pulse_idx[...,xp.newaxis], num_range_gates, axis=-1)

    samples_for_lags = xp.take(samples[...,xp.newaxis,:,:], lags_pulse_idx)

    return pulses_as_samples, samples_for_lags

def create_blanking_mask(num_range_gates, pulses_as_samples, samples_for_lags, data_mask, num_averages):

    # first do tx blanked lags
    blanked_1 = data_fitting.Einsum.transpose(pulses_as_samples)[...,xp.newaxis,xp.newaxis,:,:]
    blanked_2 = blanked_1 + 1

    x = blanked_1 == samples_for_lags[...,xp.newaxis]
    y = blanked_2 == samples_for_lags[...,xp.newaxis]

    blanking_mask = xp.any(x | y, axis=-1)

    # Blank where data does not exist
    blanking_mask[~data_mask] = True
    blanking_mask[num_averages<=0] = True

    blanking_mask = blanking_mask.transpose(0,3,2,1)

    # cut alt lag 0
    blanking_mask = blanking_mask[...,:-1]
    blanking_mask = blanking_mask.reshape(blanking_mask.shape[0], blanking_mask.shape[1],
                                            blanking_mask.shape[2] * blanking_mask.shape[3])

    return blanking_mask



def estimate_max_self_clutter(num_range_gates, pulses_as_samples, samples_for_lags, pwr0, lagfr, smsep, data_mask):

    pulses_as_samples = data_fitting.Einsum.transpose(pulses_as_samples)[...,xp.newaxis,xp.newaxis,:,:]
    samples_for_lags = samples_for_lags[...,xp.newaxis]

    first_range_in_samps = lagfr/smsep
    affected_ranges = (samples_for_lags -
                        pulses_as_samples -
                        first_range_in_samps[:,xp.newaxis,xp.newaxis,xp.newaxis,xp.newaxis])
    affected_ranges = affected_ranges.astype(int)

    ranges = xp.arange(num_range_gates)
    ranges_reshape = ranges[xp.newaxis,xp.newaxis,xp.newaxis,:,xp.newaxis]
    condition = ((affected_ranges <= samples_for_lags) &
                (affected_ranges < num_range_gates) &
                (affected_ranges != ranges_reshape) &
                (affected_ranges >= 0) &
                data_mask[...,xp.newaxis])


    tmp_pwr = xp.array(xp.broadcast_to(pwr0[...,xp.newaxis,xp.newaxis,:,xp.newaxis], condition.shape))

    tmp_pwr[~condition] = 0.0

    affected_pwr = xp.sqrt(tmp_pwr)

    tmp = xp.sqrt(pwr0[...,xp.newaxis,xp.newaxis,:,xp.newaxis]) * affected_pwr
    clutter_term12 = xp.einsum('...klm->...l', tmp)


    affected_pwr = affected_pwr[...,xp.newaxis]

    pwr_1 = affected_pwr[:,:,0]
    pwr_2 = data_fitting.Einsum.transpose(affected_pwr[:,:,1])
    clutter_term3 = xp.einsum('...ijk,...ikm->...i', pwr_1, pwr_2)

    clutter = clutter_term12 + clutter_term3

    return clutter

    # ranges_per_tau = mpinc / smsep
    # num_lags = samples_for_lags.shape[0]
    # cri = []
    # for i in range(num_lags):
    #     lag_ranges = []
    #     for j in range(num_range_gates):
    #         s1 = samples_for_lags[i,0,j]
    #         s2 = samples_for_lags[i,1,j]

    #         p1 = ptab[(ptab * ranges_per_tau) <= s1]
    #         p2 = ptab[(ptab * ranges_per_tau) <= s2]

    #         r1 = s1 - (p1*ranges_per_tau) - (lagfr/smsep)
    #         r1 = r1[r1<num_range_gates]
    #         r1 = r1[r1>0]
    #         r1 = r1[r1 != j]
    #         r1 = r1.astype(int)

    #         r2 = s2 - (p2*ranges_per_tau) - (lagfr/smsep)
    #         r2 = r2[r2<num_range_gates]
    #         r2 = r2[r2>0]
    #         r2 = r2[r2 != j]
    #         r2 = r2.astype(int)


    #         if r1.size > 0:
    #             term1 = xp.sum(xp.sqrt(pwr0[:,j][...,xp.newaxis] * pwr0[:,r1]),axis=1)

    #         else:
    #             term1 = xp.zeros(pwr0.shape[0])

    #         if r2.size > 0:
    #             term2 = xp.sum(xp.sqrt(pwr0[:,j][...,xp.newaxis] * pwr0[:,r2]),axis=1)
    #         else:
    #             term2 = xp.zeros(pwr0.shape[0])

    #         if r1.size > 0 and r2.size > 0:
    #             term3 = xp.einsum('ij,ik->i', xp.sqrt(pwr0[:,r1]), xp.sqrt(pwr0[:,r2]))
    #         else:
    #             term3 = xp.zeros(pwr0.shape[0])

    #         c = term1 + term2 + term3
    #         lag_ranges.append(c)
    #     cri.append(lag_ranges)



    # x = xp.array(cri)
    # print(x[:,1,0])
    # print(clutter[0,:,1])





def calculate_t(lags, mpinc):
    # cut alt lag 0
    lags = lags[:,:-1,1] - lags[:,:-1,0]

    # Need braces to enforce data type. Lags is short and will overflow.
    t = lags * (mpinc[...,xp.newaxis] * 1e-6)
    t = t[:, xp.newaxis,:]

    return t

def calculate_wavelength(tfreq):
    C = 299792458
    wavelength = C/(tfreq * 1e3)
    wavelength = wavelength[:,xp.newaxis,xp.newaxis]

    return wavelength

def calculate_constants(wavelength, t):

    W_constant = (-1 * 2 * xp.pi * t)/wavelength
    W_constant = W_constant[:,:,xp.newaxis,:]


    V_constant = (1j * 4 * xp.pi * t)/wavelength
    V_constant = V_constant[:,:,xp.newaxis,:]


    return {'W' : W_constant, 'V' : V_constant}

def calculate_initial_W(array_shape):
    W = xp.ones(array_shape) * 200.0

    return W

def calculate_initial_V(wavelength, num_velocity_models, num_range_gates, mpinc):
    nyquist_v = wavelength/(4.0 * mpinc[...,xp.newaxis,xp.newaxis] * 1e-6)

    step = 2 * nyquist_v / num_velocity_models
    tmp = xp.arange(num_velocity_models, dtype=xp.float64)[xp.newaxis,xp.newaxis,:]
    tmp = tmp * step
    nyquist_v_steps = (-1 * nyquist_v) + tmp

    nyquist_v_steps = nyquist_v_steps[...,xp.newaxis]

    return nyquist_v_steps


def compute_model_and_derivatives(params, **kwargs):

    p0 = params['p0']['values']
    W = params['W']['values']
    V = params['V']['values']

    model_constant_W = kwargs['W']
    model_constant_V = kwargs['V']

    def calculate_model():
        model = p0 * xp.exp(model_constant_W * W) * xp.exp(model_constant_V * V)
        return model

    model = calculate_model()
    J = xp.repeat(model[...,xp.newaxis], 3, axis=-1)

    def compute_J():
        J[:,:,:,:,0] /= p0
        J[:,:,:,:,1] *= model_constant_W
        J[:,:,:,:,2] *= model_constant_V

        return J

    model_dict = {}
    model = xp.concatenate((model.real, model.imag), axis=-1)
    model_dict['model'] = model


    J_model = compute_J()
    J_model = xp.concatenate((J_model.real, J_model.imag), axis=-2)
    model_dict['J'] = J_model

    return model_dict

def fit_all_records(records_data):
    num_velocity_models = 30
    step = 10

    params = {'p0': {'values' : None, 'min' : 0.0001, 'max' : None},
              'W': {'values' : None, 'min' : -200.0, 'max' : 8000.0},
              'V': {'values' : None, 'min' : None, 'max' : None}}


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

    data_mask = xp.array(records_data["data_mask"])

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

    initial_V = calculate_initial_V(wavelength, num_velocity_models, num_range_gates, mpinc)
    initial_W = calculate_initial_W(initial_V.shape)
    initial_p0 = pwr0[...,xp.newaxis,xp.newaxis]

    total_records = pwr0.shape[0]
    consistent_shape = (total_records, num_range_gates, num_velocity_models, 1)
    reshaper = xp.zeros(consistent_shape)

    initial_V = initial_V + reshaper
    initial_W = initial_W + reshaper
    initial_p0 = initial_p0 + reshaper

    even_chunks = total_records // step
    remainder = total_records - (even_chunks * step)

    fits = []
    def do_fits(args):

        start = args[0]
        stop = args[1]

        W_i = initial_W[start:stop,...]
        V_i = initial_V[start:stop,...]
        p0_i = initial_p0[start:stop,...]

        V_upper_bound = xp.repeat(initial_V[start:stop,:,-1,xp.newaxis], num_velocity_models, axis=2)
        V_lower_bound = xp.repeat(initial_V[start:stop,:,0,xp.newaxis], num_velocity_models, axis=2)

        params['p0']['values'] = p0_i
        params['W']['values'] = W_i
        params['V']['values'] = V_i

        params['V']['max'] = V_upper_bound
        params['V']['min'] = V_lower_bound


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
            argv.append((i*step,(i+1)*step))

    if remainder:
        argv.append(even_chunks*step, total_records)

    p = ThreadPool()

    fits = p.map(do_fits, argv)

    tmp = ([], [], [], [], [])

    for f in fits:
        tmp[0].append(f.fitted_params['p0']['values'])
        tmp[1].append(f.fitted_params['W']['values'])
        tmp[2].append(f.fitted_params['V']['values'])
        tmp[3].append(f.cov_mat)
        tmp[4].append(f.converged)

    fitted_data = {'p0' : xp.vstack(tmp[0]),
                   'W' : xp.vstack(tmp[1]),
                   'V' : xp.vstack(tmp[2]),
                   'cov_mat' : xp.vstack(tmp[3]),
                   'converged' : xp.vstack(tmp[4])}


    return fitted_data

def write_to_file(records_data, fitted_data, output_name):

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

















