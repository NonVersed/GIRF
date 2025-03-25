# Python code to generate a chirp waveform for ST4 GIRF calibration.
# Chirp waveform is generated on a dwell time of 10.
# For most accurate GIRF estimation, the waveform is programmed with zero-padding on left and right sides and
# with long ramp-up times.

import numpy as np
import scipy
import sigpy as sp
import sigpy.plot as pl
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('./scripts')
import mda_io as mi

if __name__ == '__main__':
    # duration of chirp (us)
    duration = 12800
    ramp_time_1 = 500
    ramp_time_2 = 500
    pad_time_1 = 500
    pad_time_2 = 500

    # sampling time of chirp (us)
    dwell_time = 10
    # starting and ending frequency of chirp (MHz or cycles per unit)
    f1 = 0
    f2 = 0.02
    # set times
    times = np.arange(0, duration-ramp_time_1-ramp_time_2-pad_time_1-pad_time_2+1, 1)

    # generate chirp
    chirp = scipy.signal.chirp(times, f1, duration, f2)

    # add ramps to chirp
    ramp_1 = np.linspace(0, chirp[0], ramp_time_1)
    ramp_2 = np.linspace(chirp[len(chirp)-1], 0, ramp_time_2)
    chirp = np.concatenate([ramp_1, chirp, ramp_2])

    # add pads to chirp
    chirp = np.pad(chirp, (pad_time_1, pad_time_2), mode='constant')


    # sample chirp
    chirp_sampled = chirp[::dwell_time]

    # write chirp to text file
    fileName = 'chirp_20kHz_v2'
    np.savetxt(os.path.join('waveforms', fileName + '.txt'), chirp_sampled, fmt='%.7f', newline='\n')

    # write chirp to mda file
    mi.writemda(os.path.join('waveforms', fileName + '.mda'), chirp_sampled)







