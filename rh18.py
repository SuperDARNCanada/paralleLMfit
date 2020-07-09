import numpy as np
import rawacf_dmap_read as rdr
import data_fitting
import sys

def determine_noise(pwr0):

    sorted_pwr0 = np.sort(pwr0)
    noise = np.mean(sorted_pwr0[:,0:10], axis=1)
    return noise

def first_order_weights(pwr0, noise, clutter, nave, blanking_mask):

    total_signal = (pwr0[:,np.newaxis,:] + noise[:,np.newaxis,np.newaxis] + clutter)

    error = total_signal / np.sqrt(nave[:,np.newaxis,np.newaxis])
    error = np.repeat(error[:,:,np.newaxis,:], blanking_mask.shape[1], axis=2)

    mask = np.repeat(blanking_mask[np.newaxis,...], error.shape[0], axis=0)


    error[mask] = 1e50
    error = np.einsum('ijkl->ilkj', error)
    error = error.reshape((error.shape[0], error.shape[1], error.shape[2] * error.shape[3]))

    weights_shape = (error.shape[0], error.shape[1], error.shape[2], error.shape[2])
    weights = np.zeros(weights_shape)

    diag = np.arange(error.shape[-1])
    weights[:,:,diag,diag] = 1


    return weights


def calculate_samples(num_range_gates, ltab, ptab, mpinc, lagfr, smsep):

    ranges = np.arange(num_range_gates)
    pulses_by_ranges = np.repeat(ptab[:,np.newaxis], num_range_gates, axis=1)

    ranges_per_tau = mpinc / smsep

    pulses_as_samples = (pulses_by_ranges * ranges_per_tau)
    first_range_in_samps = lagfr/smsep
    samples = (pulses_by_ranges * ranges_per_tau) + ranges[np.newaxis,:] + first_range_in_samps

    lags = ltab[0:-1,1] - ltab[0:-1,0]
    lag_pairs = ltab[0:-1, :]
    lags_pulse_idx = np.searchsorted(ptab, lag_pairs)

    samples_for_lags = samples[lags_pulse_idx]

    return pulses_as_samples, samples_for_lags

def find_tx_blanked_lags(num_range_gates, pulses_as_samples, samples_for_lags):

    blanked_1 = pulses_as_samples.T[np.newaxis,np.newaxis,...]
    blanked_2 = blanked_1 + 1

    x = blanked_1 == samples_for_lags[...,np.newaxis]
    y = blanked_2 == samples_for_lags[...,np.newaxis]

    blanking_mask = np.any(np.logical_or(x,y), axis=-1)

    return blanking_mask



def estimate_max_self_clutter(num_range_gates, pulses_as_samples, samples_for_lags, pwr0, lagfr, smsep):

    pulses_as_samples = pulses_as_samples.T[np.newaxis,np.newaxis,...]
    first_range_in_samps = lagfr/smsep
    affected_ranges = samples_for_lags[...,np.newaxis] - pulses_as_samples - first_range_in_samps
    affected_ranges = affected_ranges.astype(int)

    ranges = np.arange(num_range_gates)
    ranges_reshape = ranges[np.newaxis,np.newaxis,:,np.newaxis]
    condition = np.logical_and((affected_ranges <= samples_for_lags[...,np.newaxis]),
                    (np.logical_and((affected_ranges < num_range_gates),
                    np.logical_and((affected_ranges != ranges_reshape), (affected_ranges >= 0)))))

    filtered_ranges = np.where(condition, affected_ranges, -1)

    tmp_pwr = np.zeros((pwr0.shape[0], pwr0.shape[1] + 1))
    tmp_pwr[:,0:-1] += pwr0

    affected_pwr = tmp_pwr[:,filtered_ranges.flatten()]
    affected_pwr = affected_pwr.reshape((affected_pwr.shape[0],) + filtered_ranges.shape)
    affected_pwr = np.sqrt(affected_pwr)

    tmp = np.sqrt(pwr0[:,np.newaxis,np.newaxis,:,np.newaxis]) * affected_pwr
    clutter_term12 = np.einsum('...klm->...l', tmp)


    affected_pwr = affected_pwr[...,np.newaxis]
    clutter_term3 = np.einsum('...ijk,...ikm->...i', affected_pwr[:,:,0], np.einsum('...lm->...ml',affected_pwr[:,:,1]))


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
    #             term1 = np.sum(np.sqrt(pwr0[:,j][...,np.newaxis] * pwr0[:,r1]),axis=1)

    #         else:
    #             term1 = np.zeros(pwr0.shape[0])

    #         if r2.size > 0:
    #             term2 = np.sum(np.sqrt(pwr0[:,j][...,np.newaxis] * pwr0[:,r2]),axis=1)
    #         else:
    #             term2 = np.zeros(pwr0.shape[0])

    #         if r1.size > 0 and r2.size > 0:
    #             term3 = np.einsum('ij,ik->i', np.sqrt(pwr0[:,r1]), np.sqrt(pwr0[:,r2]))
    #         else:
    #             term3 = np.zeros(pwr0.shape[0])

    #         c = term1 + term2 + term3
    #         lag_ranges.append(c)
    #     cri.append(lag_ranges)



    # x = np.array(cri)
    # print(x[:,1,0])
    # print(clutter[0,:,1])





def calculate_t(lags, mpinc):
    lags = lags[0:-1,1] - lags[0:-1,0]

    # Need braces to enforce data type. Lags is short and will overflow.
    t = lags * (mpinc * 1e-6)
    t = t[np.newaxis, np.newaxis,:]

    return t

def calculate_wavelength(tfreq):
    C = 299792458
    wavelength = C/(tfreq * 1e3)
    wavelength = wavelength[:,np.newaxis,np.newaxis]

    return wavelength

def calculate_constants(wavelength, t):
    W_constant = (-1 * 2 * np.pi * t)/wavelength
    W_constant = W_constant[:,:,np.newaxis,:]


    V_constant = (1j * 4 * np.pi * t)/wavelength
    V_constant = V_constant[:,:,np.newaxis,:]


    return {'W' : W_constant, 'V' : V_constant}

def calculate_initial_W(array_shape):
    W = np.ones(array_shape) * 200.0

    return W

def calculate_initial_V(wavelength, num_velocity_models, num_range_gates, mpinc):
    nyquist_v = wavelength/(4.0 * mpinc * 1e-6)

    step = 2 * nyquist_v / num_velocity_models
    tmp = np.arange(num_velocity_models, dtype=np.float64)[np.newaxis,np.newaxis,:]
    tmp = tmp * step
    nyquist_v_steps = (-1 * nyquist_v) + tmp

    nyquist_v_steps = nyquist_v_steps[...,np.newaxis]

    return nyquist_v_steps


def calculate_weights(num_records, num_range_gates, num_velocity_models, num_lags):

    weights_real = np.zeros((num_records, num_range_gates, num_velocity_models, num_lags, num_lags))
    weights_real[...,np.arange(num_lags),np.arange(num_lags)] = 1

    weights_imag = np.zeros((num_records, num_range_gates, num_velocity_models, num_lags, num_lags))
    weights_imag[...,np.arange(num_lags),np.arange(num_lags)] = 1

    return {'real' : weights_real, 'imag' : weights_imag}

def compute_model_and_derivatives(params, **kwargs):

    p0 = params['p0']['values']
    W = params['W']['values']
    V = params['V']['values']

    model_constant_W = kwargs['W']
    model_constant_V = kwargs['V']

    def calculate_model():
        model = p0 * np.exp(model_constant_W * W) * np.exp(model_constant_V * V)
        return model

    model = calculate_model()
    J = np.repeat(model[:,:,:,:,np.newaxis], 3, axis=4)

    def compute_J():
        J[:,:,:,:,0] /= p0
        J[:,:,:,:,1] *= model_constant_W
        J[:,:,:,:,2] *= model_constant_V

        return J

    model_dict = {}
    model_dict['model'] = np.concatenate((model.real, model.imag), axis=-1)

    J_model = compute_J()
    J_model = np.concatenate((J_model.real, J_model.imag), axis=-2)
    model_dict['J'] = J_model

    return model_dict

def fit_all_records(records_data):
    num_velocity_models = 30
    step = 10

    params = {'p0': {'values' : None, 'min' : 0.0001, 'max' : None},
              'W': {'values' : None, 'min' : -200.0, 'max' : None},
              'V': {'values' : None, 'min' : None, 'max' : None}}

    for k,v in records_data.items():
        data_for_fitting = v['split_data']

        transmit_freq = np.array(data_for_fitting['tfreq'])
        offset = np.array(data_for_fitting['offset'])
        num_averages = np.array(data_for_fitting['nave'])
        pwr0 = np.array(data_for_fitting['pwr0'])
        acf = np.array(data_for_fitting['acfd'])
        xcf = np.array(data_for_fitting['xcfd'])
        lags = np.array(data_for_fitting['ltab'])
        pulses = np.array(data_for_fitting['ptab'])

        mpinc = data_for_fitting['mpinc']
        lagfr = data_for_fitting['lagfr']
        smsep = data_for_fitting['smsep']
        num_range_gates = data_for_fitting['nrang']
        num_lags = data_for_fitting['acfd'].shape[2]


        acf = np.einsum('...ij->...ji', acf)

        pwr0[...] = 1.0
        transmit_freq[...] = 10500
        acf[...,0,:] = 1.0
        acf[...,1,:] = 0.0

        acf = acf.reshape((acf.shape[0], acf.shape[1], acf.shape[2] * acf.shape[3]))
        t = calculate_t(lags, mpinc)
        wavelength = calculate_wavelength(transmit_freq)


        noise = determine_noise(pwr0)


        pulses_as_samples, samples_for_lags = calculate_samples(num_range_gates, lags, pulses,
                                                                    mpinc, lagfr, smsep)
        blanking_mask = find_tx_blanked_lags(num_range_gates, pulses_as_samples, samples_for_lags)
        clutter = estimate_max_self_clutter(num_range_gates, pulses_as_samples, samples_for_lags,
                                            pwr0, lagfr, smsep)
        fo_weights = first_order_weights(pwr0, noise, clutter, num_averages, blanking_mask)

        model_constants = calculate_constants(wavelength, t)

        initial_V = calculate_initial_V(wavelength, num_velocity_models, num_range_gates, mpinc)
        initial_W = calculate_initial_W(initial_V.shape)
        initial_p0 = pwr0[:,:,np.newaxis,np.newaxis] * 2

        total_records = len(v['record_nums'])
        consistent_shape = (total_records, num_range_gates, num_velocity_models, 1)
        reshaper = np.zeros(consistent_shape)

        initial_V = initial_V + reshaper
        initial_W = initial_W + reshaper
        initial_p0 = initial_p0 + reshaper


        even_chunks = total_records // step
        remainder = total_records - (even_chunks * step)

        fits = []
        def do_fits(start, stop):

            W_i = initial_W[start:stop,...]
            V_i = initial_V[start:stop,...]
            p0_i = initial_p0[start:stop,...]

            V_upper_bound = np.repeat(initial_V[start:stop,:,-1,np.newaxis], num_velocity_models, axis=2)
            V_lower_bound = np.repeat(initial_V[start:stop,:,0,np.newaxis], num_velocity_models, axis=2)

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

            lm_step_shape = (step, num_range_gates, num_velocity_models, 1)

            fits.append(data_fitting.LMFit(acf_i, model_dict, weights, params, lm_step_shape))


        if even_chunks > 0:
            for i in range(even_chunks-1):
                do_fits(i*step,(i+1)*step)

        if remainder:
            do_fits(even_chunks*step, total_records)

        fits = np.array(fits).reshape(-1,*fits.shape[2:])
        records_data[k]['fitted_data'] = fits


if __name__ == '__main__':
    data_read = rdr.RawacfDmapRead("20170714.0601.00.sas.rawacf")

    records_data = data_read.get_parsed_data()

    fit_all_records(records_data)











