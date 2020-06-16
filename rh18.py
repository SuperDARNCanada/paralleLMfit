import numpy as np
import rawacf_dmap_read as rdr
from data_fitting import fit_data
import sys

def estimate_max_self_clutter(num_range_gates, ltab, ptab, mpinc, pwr0, lagfr, smsep):
    first_range_in_samps = lagfr/smsep
    lags = ltab[0:-1,1] - ltab[0:-1,0]

    lag_pairs = ltab[0:-1, :]

    num_lags = lags.shape[0]
    num_pulses = ptab.shape[0]

    ranges_per_tau = mpinc / smsep

    time_to_first_range = lagfr / 2

    ranges = np.arange(num_range_gates)
    pulses_by_ranges = np.repeat(ptab[:,np.newaxis], num_range_gates, axis=1)
    samples = (pulses_by_ranges * ranges_per_tau) + ranges[np.newaxis,:] + (lagfr/smsep)


    lags_pulse_idx = np.searchsorted(ptab, lag_pairs)
    samples_for_lags = samples[lags_pulse_idx]

    pulse_as_sample = (pulses_by_ranges * ranges_per_tau).T[np.newaxis,np.newaxis,...]
    affected_ranges = samples_for_lags[...,np.newaxis] - pulse_as_sample - (lagfr/smsep)
    affected_ranges = affected_ranges.astype(int)


    ranges_reshape = ranges[np.newaxis,np.newaxis,:,np.newaxis]
    condition = (np.logical_and((affected_ranges < num_range_gates),
                    np.logical_and((affected_ranges == ranges_reshape), (affected_ranges >= 0))))

    filtered_ranges = np.where(condition, affected_ranges, -1)

    tmp_pwr = np.zeros((pwr0.shape[0], pwr0.shape[1] + 1))
    tmp_pwr[:,0:-1] += pwr0

    affected_pwr = tmp_pwr[:,filtered_ranges.flatten()]
    affected_pwr = affected_pwr.reshape((affected_pwr.shape[0],) + filtered_ranges.shape)

    tmp = np.sqrt(pwr0[:,np.newaxis,np.newaxis,:,np.newaxis] * affected_pwr)
    cri_term12 = np.einsum('...klm->...l', tmp)


    cri_term3 = np.einsum('...kl,...lk->...k', affected_pwr[:,:,0,:,:], np.einsum('...lm->...ml',affected_pwr[:,:,1,:,:]))

    cri = cri_term12 + cri_term3

    # pwr0_reshape = np.repeat(tmp_pwr[:,np.newaxis,np.newaxis,np.newaxis,:], tmp_pwr.shape[-1], axis=3)
    # pwr0_reshape = np.repeat(pwr0_reshape, )
    # print(pwr0_reshape.shape, filtered_ranges.shape)


    # affected_pwr = np.zeros(pwr0_reshape.shape)

    # blah = pwr0_reshape[filtered_ranges[np.newaxis,...]]
    # print(blah.shape)
    # print(blah[0,0,0,0,0,0,0,100,:])
    # affected_pwr += pwr0_reshape[...,0:-1,0:-1][filtered_ranges[np.newaxis,...]]


    # print(affected_pwr)




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
    #             term1 = np.sum(np.sqrt(pwr0[:,j] * pwr0[:,r1]),axis=1)
    #         else:
    #             term1 = np.zeros(pwr0.shape[0])

    #         if r2.size > 0:
    #             term2 = np.sum(np.sqrt(pwr0[:,j] * pwr0[:,r2]),axis=1)
    #         else:
    #             term2 = np.zeros(pwr0.shape[0])

    #         if r1.size > 0 and r2.size > 0:
    #             term3 = np.einsum('ij,ik->i', pwr0[:,r1], pwr0[:,r2])
    #         else:
    #             term3 = np.zeros(pwr0.shape[0])

    #         cri.append(term1 + term2 + term3)

    # print(cri[1])



            #lag_ranges.append((r1,r2))



        #ranges_for_lags.append(lag_ranges)

    #print(ranges_for_lags)






    #ranges = np.repeat(ranges[np.newaxis,np.newaxis,np.newaxis,:], )
    #print(ppr[np.newaxis,np.newaxis,...] <= samples_for_lags)
    # interfering_pwr = pwr0[:,np.newaxis,np.newaxis,np.newaxis,:]
    # interfering_pwr = np.repeat(interfering_pwr, num_lags, axis=1)
    # interfering_pwr = np.repeat(interfering_pwr, 2, axis=2)
    # interfering_pwr = np.repeat(interfering_pwr, num_pulses, axis=3)

    # interfering_pwr[:,ppr[np.newaxis,np.newaxis,...] <= samples_for_lags] = 0

    # print(interfering_pwr[0,0,0,:,0])
    # samples_for_lags[ppr[np.newaxis,np.newaxis,...] <= samples_for_lags] = 0

    # print(samples_for_lags[0,0,:,0])





    #a = (ptab[:,np.newaxis] * ranges_per_tau) <= samples
    #print((ptab[:,np.newaxis] * ranges_per_tau))
    #print(samples)
    # truth = (samples[] >= (pulses_by_ranges * ranges_per_tau))
    # p1 = np.where((samples >= (pulses_by_ranges * ranges_per_tau)))[1]

    #print(p1)
    #lags_pulse_idx = np.searchsorted(ptab, lag_pairs)

    # p1
    # s2 = samples[lags_pulse_idx]

    # print(s1.shape, s2.shape)




def calculate_t(lags, mpinc):
    lags = lags[0:-1,1] - lags[0:-1,0]
    t = lags * mpinc * 1e-6
    t = t[np.newaxis, np.newaxis,:]

    return t

def calculate_wavelength(tfreq):
    C = 299792458
    wavelength = C/(tfreq * 1e3)
    wavelength = wavelength[:,np.newaxis,np.newaxis]

    return wavelength

def calculate_constants(wavelength, t):
    real_constant = (-1 * 2 * np.pi * t)/wavelength
    real_constant = real_constant[:,:,np.newaxis,:]

    imag_constant = (1j * 4 * np.pi * t)/wavelength
    imag_constant = imag_constant[:,:,np.newaxis,:]

    return {'real' : real_constant, 'imag' : imag_constant}

def calculate_initial_W(num_records, num_range_gates, num_velocity_models, num_lags):
    W = np.ones((num_records, num_range_gates, num_velocity_models, num_lags)) * 200.0

    return W

def calculate_initial_V(wavelength, num_velocity_models, num_lags, mpinc):
    nyquist_v = wavelength/(4.0 * mpinc * 1e-6)
    step = 2 * nyquist_v / num_velocity_models
    tmp = np.arange(num_velocity_models, dtype=np.float64)[np.newaxis,np.newaxis,:]
    tmp = tmp * step
    nyquist_v_steps = (-1 * nyquist_v) + (tmp * step)
    nyquist_v_steps = np.repeat(nyquist_v_steps[...,np.newaxis], num_lags, axis=-1)

    return nyquist_v_steps



def calculate_initial_p0(pwr0):
    return pwr0[:,:,np.newaxis,np.newaxis]


def calculate_weights(num_records, num_range_gates, num_velocity_models, num_lags):

    weights_real = np.zeros((num_records, num_range_gates, num_velocity_models, num_lags, num_lags))
    weights_real[...,np.arange(num_lags),np.arange(num_lags)] = 1

    weights_imag = np.zeros((num_records, num_range_gates, num_velocity_models, num_lags, num_lags))
    weights_imag[...,np.arange(num_lags),np.arange(num_lags)] = 1

    return {'real' : weights_real, 'imag' : weights_imag}

def fit_all_records(records_data):
    for k,v in records_data.items():
        data_for_fitting = v['split_data']

        mpinc = data_for_fitting['mpinc']
        lags = data_for_fitting['ltab']

        pulses = data_for_fitting['ptab']

        t = calculate_t(lags, mpinc)

        wavelength = calculate_wavelength(data_for_fitting['tfreq'])

        model_constants = calculate_constants(wavelength, t)

        total_records = len(v['record_nums'])

        even_chunks = total_records // 20
        remainder = total_records - (even_chunks * 20)

        fits = []
        def do_fits(start, stop):
            wavelength = calculate_wavelength(data_for_fitting['tfreq'][start:stop])

            model_constants = calculate_constants(wavelength, t)

            acf_real = data_for_fitting['acfd'][start:stop,...,0][...,np.newaxis,:]
            acf_imag = data_for_fitting['acfd'][start:stop,...,1][...,np.newaxis,:]

            acf = {'real' : acf_real, 'imag' : acf_imag}

            num_records = acf_real.shape[0]
            num_range_gates = acf_real.shape[1]
            num_velocity_models = 30
            num_lags = data_for_fitting['mplgs']

            estimate_max_self_clutter(num_range_gates, lags, pulses, mpinc, data_for_fitting['pwr0'], data_for_fitting['lagfr'], data_for_fitting['smsep'])


            initial_W = calculate_initial_W(num_records, num_range_gates, num_velocity_models, num_lags)
            initial_V = calculate_initial_V(wavelength[start:stop,...], num_velocity_models, num_lags, mpinc)
            initial_p0 = calculate_initial_p0(data_for_fitting['pwr0'][start:stop])

            model_params = {'p0' : initial_p0, 'W' : initial_W, 'V' : initial_V}

            weights = calculate_weights(num_records, num_range_gates, num_velocity_models, num_lags)

            fits.append(fit_data(model_params, model_constants, acf, weights))

        if even_chunks > 0:
            for i in range(even_chunks-1):
                do_fits(i*20,(i+1)*20)

        if remainder:
            do_fits(even_chunks*20, total_records)

        fits = np.array(fits).reshape(-1,*fits.shape[2:])
        records_data[k]['fitted_data'] = fits



if __name__ == '__main__':
    data_read = rdr.RawacfDmapRead("20170714.0601.00.sas.rawacf")

    records_data = data_read.get_parsed_data()

    fit_all_records(records_data)











