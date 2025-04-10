#!/usr/bin/env python

# Utilize a wavelet basis to model a BBH signal merger.
# Based on the work of Finch and Moore (https://arxiv.org/abs/2108.09344,
# https://arxiv.org/abs/2205.07809)
# Author: Alex Correia

import numpy as np
import math as m
from pycbc.types import TimeSeries, zeros
from pycbc.waveform import NoWaveformError

pi = np.pi

def parse_params(**kwargs):
    """Generate dictionaries for each wavelet's parameters.
    Checks if the minimum required parameters are provided.
    """
    # number of wavelets
    try:
        w = int(kwargs['wavelets'])
    except KeyError:
        raise ValueError('Must provide number of wavelets to generate')

    amps = {}
    freqs = {}
    taus = {}
    phis = {}
    etas = {}

    for i in range(w):
        s = str(int(i+1))
        # amplitude
        try:
            amps[s] = kwargs['amp' + s]
        except KeyError:
            raise ValueError(f'missing amp{s}')

    	# frequencies
        try:
            freqs[s] = kwargs['freq' + s]
        except KeyError:
            raise ValueError(f'missing freq{s}')

    	# damping times
        try:
            taus[s] = kwargs['tau' + s]
        except KeyError:
            raise ValueError(f'missing tau{s}')

    	# phases
        try:
            phis[s] = kwargs['phi' + s]
        except KeyError:
            raise ValueError(f'missing phi{s}')

    	# ref times
        try:
            etas[s] = kwargs['eta' + s]
        except KeyError:
            raise ValueError(f'missing eta{s}')

    return w, amps, freqs, taus, phis, etas

def get_td_wavelet(amp, phi, f, tau, eta, start_time, end_time, dt):
    r"""Generate a single wavelet in the time domain.
        This uses the Morlet-Gabot formula as listed in arXiv:2108.09344:

	.. math::

	   h(t) &:= h_{+} + ih_{\cross} \\
		&:= \sum_{w=1}^W A_w \exp \Big[-2\pi i\nu_w (t-\eta_w) \\
		& - \big( \frac{t-\eta_w}{\tau_w} \big)^2 - i\phi_w \Big], t_i < t < t_f

    Parameters
    ----------
    amp : float
        The wavelet amplitude.

    phi : float
        The wavelet phase.

    f : float
        The wavelet frequency in Hz.

    tau : float
        The wavelet damping time in seconds.

    eta : float
        The central time in seconds of the wavelet. This time corresponds to:

	.. math::

	   h_w(t = \eta_w) = A_w\exp (i \phi_w)

    start_time : float
        The start time in seconds of the wavelet.

    end_time : float
        The end time in seconds of the wavelet.

    dt : float
        The sample time in seconds of the waveform.

    Returns
    -------
    (array, array)
        The time domain plus and cross polarizations of the wavelet.
    """    
    # generate a time series for the wavelet
    l = m.ceil((end_time - start_time)/dt)
    t = np.linspace(start_time, end_time, l)

    # evaluate the wavelet
    offset = t - eta
    nondim_offset = offset/tau
    wf = amp*np.exp(-2*pi*1j*f*offset - nondim_offset*nondim_offset + 1j*phi)

    # retrieve the plus and cross polarizations
    hp = wf.real
    hc = -wf.imag
    
    # convert to time series
    hp = TimeSeries(hp, delta_t=dt)
    hc = TimeSeries(hc, delta_t=dt)
    return hp, hc


def wavelet_sum_base(input_params, sum_basis=True):
    """Base function for returning a superposition of wavelets.

    Parameters
    ----------
    input_params : dict
        	Dictionary of parameters for generating wavelets. See
        get_td_wavelet for list of params.
    sum_basis : bool, optional
        Flag whether to sum together the wavelets. If False, return the
        individual wavelets. If True (default), return the sum of wavelets.
    """
    # parse parameters
    w, amps, freqs, taus, phis, etas = parse_params(**input_params)
    assert w > 0, "Must generate at least one wavelet in wavelet basis"

    t_start = input_params['t_start']
    t_end = input_params['t_end']
    dt = input_params['delta_t']

    # allocate hp, hc vectors using the length of the segment
    tlen = t_end - t_start
    if tlen < dt:
        raise NoWaveformError('Length of wavelet is less than one sample. ' +
                              'Try decreasing start or increasing end time.')
    ilen = m.ceil(tlen/dt)
    if sum_basis:
        hp_out = TimeSeries(zeros(ilen, dtype=np.float64), delta_t=dt)
        hc_out = TimeSeries(zeros(ilen, dtype=np.float64), delta_t=dt)
    else:
        out = {}

    # generate wavelets and add to out vectors
    for i in range(w):
        s = str(int(i+1))
        hp, hc = get_td_wavelet(amps[s], phis[s], freqs[s], taus[s], etas[s],
                                t_start, t_end, dt)
        if sum_basis:
            hp_out += hp
            hc_out += hc
        else:
            hp.start_time = -tlen
            hc.start_time = -tlen
            out[s] = [hp, hc]

    ### set the end time as the zero point (assuming end time is tc for
    ### inspiral-merger models)
    if sum_basis:
        hp_out.start_time = -tlen
        hc_out.start_time = -tlen
        return hp_out, hc_out
    else:
        return out

### Approximants ##############################################################

def get_td_wavelet_basis(**kwargs):
    """Generate a sum of wavelets in the time domain.
    """
    return wavelet_sum_base(kwargs)

def get_td_wavelets(**kwargs):
    """Generate time domain wavelets.
    """
    return wavelet_sum_base(kwargs, sum_basis=False)
