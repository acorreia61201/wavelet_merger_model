#!/usr/bin/env python

# Utilize a wavelet basis to model a BBH signal merger.
# Based on the work of Finch and Moore (https://arxiv.org/abs/2108.09344,
# https://arxiv.org/abs/2205.07809)
# Author: Alex Correia

import numpy as np
import math as m
from pycbc.types import TimeSeries, zeros
from pycbc.waveform import NoWaveformError
import warnings

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
    extra_args = {}

    # extra args for signal length, tapering
    extra_args['wavelet_ref_index'] = int(kwargs.get('wavelet_ref_index', 1))
    if extra_args['wavelet_ref_index'] > w:
        raise KeyError('Reference index exceeds number of wavelets')
    extra_args['wavelet_tau_duration'] = kwargs.get('wavelet_tau_duration', 5)
    extra_args['wavelet_max_duration'] = kwargs.get('wavelet_max_duration')
    extra_args['wavelet_taper'] = kwargs.get('wavelet_taper')
    extra_args['wavelet_taper_duration'] = kwargs.get('wavelet_taper_duration',
                                                      0)

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
        if int(s) == int(extra_args['wavelet_ref_index']):
            if 'eta' + s in kwargs.keys():
                warnings.warn(f'Wavelet index {s} has a user input but is also '
                              f'defined as reference. Setting eta{s} to zero')
            kwargs['eta' + s] = 0
        try:
            print(kwargs['eta' + s])
            etas[s] = kwargs['eta' + s]
        except KeyError:
            raise ValueError(f'missing eta{s}')

    return w, amps, freqs, taus, phis, etas, extra_args

def get_td_wavelet(amp, phi, f, tau, eta, duration, dt):
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
        The central time in seconds of the wavelet, defined relative to zero 
        (the coalescence time in the detector frame). This time corresponds to:

	.. math::

	   h_w(t = \eta_w) = A_w\exp (i \phi_w)

    duration : float
        The duration in seconds over which to generate the wavelet.
    dt : float
        The sample time in seconds of the waveform.

    Returns
    -------
    (array, array)
        The time domain plus and cross polarizations of the wavelet.
    """    
    # generate a time series for the wavelet
    l = m.ceil(duration/dt)
    t = np.linspace(-duration/2, duration/2, l)

    # evaluate the wavelet
    offset = t - eta
    nondim_offset = offset/tau
    wf = amp*np.exp(-2*pi*1j*f*offset - nondim_offset*nondim_offset - 1j*phi)

    # retrieve the plus and cross polarizations
    hp = wf.real
    hc = wf.imag
    
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
        
    Extra args for signal output (passed via input_params)
    ------------------------------------------------------
    wavelet_ref_index : str, optional
        Specifies which wavelet index is used as the reference for generating
        the signal. This sets the peak time of the reference wavelet equal to
        zero, at the center of the generated window. If an eta is provided for
        the specified reference wavelet, that input is overriden, so that the 
        reference is at zero no matter what. Default '1'.
    wavelet_tau_duration : float, optional
        The duration of the signal, as a multiple of longest provided damping
        time. Default 5.
    wavelet_max_duration : float, optional
        The maximum duration in seconds of the cumulative waveform. This
        overrides wavelet_tau_duration, i.e. if tau_duration > max_duration,
        the generated waveform will be max_duration long; otherwise, the
        waveform will be tau_duration. Default 0, which ensures the tau
        duration is used by default.
    wavelet_taper : str or None, optional
        Flag whether to taper the cumulative waveform using a Tukey window.
        Accepts 'start', 'end', 'startend', or None if not tapering. Default
        None.
    wavelet_taper_duration : float, optional
        Length in seconds of the taper window(s) if wavelet_taper is an
        accepted string. Default 0.
    """
    # parse parameters
    w, amps, freqs, taus, phis, etas, extra_args = parse_params(**input_params)
    assert w > 0, "Must generate at least one wavelet in wavelet basis"

    # determine the duration of the output wf
    dt = input_params['delta_t']
    tau_dur = extra_args['wavelet_tau_duration'] * max(taus.values())
    if extra_args['wavelet_max_duration'] is None:
        dur = tau_dur
    else:
        assert extra_args['wavelet_max_duration'] > dt, \
            ("wavelet_max_duration must be greater than one sample if "
             "specified")
        dur = min(tau_dur, extra_args['wavelet_max_duration'])
    t_start = -dur/2
        
    # catch whether duration is less than sample size
    if dur < dt:
        raise NoWaveformError('Length of waveform is less than one sample. '
                              'Try increasing duration or decreasing sample '
                              'length.')

    # allocate hp, hc vectors using the length of the segment
    ilen = m.ceil(dur/dt)
    if sum_basis:
        hp_out = TimeSeries(zeros(ilen, dtype=np.float64), delta_t=dt,
                            epoch=t_start)
        hc_out = TimeSeries(zeros(ilen, dtype=np.float64), delta_t=dt,
                            epoch=t_start)
    else:
        out = {}

    # common function for tapering the waveforms
    def taper_output(hp, loc, win_len):
        if float(win_len) > float(hp.duration):
            raise ValueError(f'Window duration {win_len} is longer than '
                             f'waveform duration {hp.duration}')
        accept = ['start', 'end', 'startend', None]
        if loc == None:
            return hp
        # FIXME: loc not in accept doesn't work; string gets read in w/ quotes
        elif 'start' not in loc and 'end' not in loc:
            raise ValueError(f"Taper argument {loc} not accepted; "
                             f"accepted args are: {accept}")
        else:
            # get length of window in samples
            idx_len = m.ceil(win_len / hp.delta_t)
            samps = np.arange(idx_len)
            # set up a cosine function for the window
            raw_taper = 1/2 * (1 - np.cos(pi*samps/idx_len))
            if 'start' in loc:
                # multiply the first samples by the raw taper
                hp[:idx_len] *= raw_taper
            if 'end' in loc:
                # multiply the last samples by the raw taper
                hp[len(hp) - idx_len:len(hp)] *= raw_taper[::-1]
            return hp

    wt = extra_args['wavelet_taper']
    wtdur = extra_args['wavelet_taper_duration']

    # generate wavelets and add to out vectors
    for i in range(w):
        s = str(int(i+1))
        # generate each wavelet with eta relative to the first central time
        hp, hc = get_td_wavelet(amps[s], phis[s], freqs[s], taus[s], 
                                etas[s], dur, dt)
        
        # set times such that t = 0 corresponds to first wavelet's peak time
        hp.start_time = t_start
        hc.start_time = t_start
        if sum_basis:
            hp_out += hp
            hc_out += hc
        else:
            # apply tapering to each wavelet
            hp = taper_output(hp, wt, wtdur)
            hc = taper_output(hc, wt, wtdur)
            out[s] = [hp, hc]

    if sum_basis:
        # apply tapering to the wavelet sum
        hp_out = taper_output(hp_out, wt, wtdur)
        hc_out = taper_output(hc_out, wt, wtdur)
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
