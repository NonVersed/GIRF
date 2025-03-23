# Python code to generate a chirp waveform for ST4 GIRF calibration.
# Chirp waveform is generated on a dwell time of 10.

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
    # sampling time of chirp (us)
    dwell_time = 10
    # starting and ending frequency of chirp (MHz or cycles per unit)
    f1 = 0
    f2 = 0.05
    # set times
    times = np.arange(0, duration+1, 1)

    # generate chirp
    chirp = scipy.signal.chirp(times, f1, duration, f2)

    # sample chirp
    chirp_sampled = chirp[::dwell_time]

    # write chirp to text file
    fileName = 'chirp_50kHz'
    np.savetxt(os.path.join('waveforms', fileName + '.txt'), chirp_sampled, fmt='%.7f', newline='\n')

    # write chirp to mda file
    mi.writemda(os.path.join('waveforms', fileName + '.mda'), chirp_sampled)

    # compare FFTs of chirp and sampled chirp
    freqs = np.linspace((-2 * 1) ** -1, (2 * 1) ** -1, len(chirp))
    chirp_fft = sp.fft(chirp)
    freqs_sampled = np.linspace((-2 * dwell_time) ** -1, (2 * dwell_time) ** -1, len(chirp_sampled))
    chirp_sampled_fft = sp.fft(chirp_sampled)

    # plot FFTs
    plt.plot(freqs, np.abs(chirp_fft))
    plt.plot(freqs_sampled, np.abs(chirp_sampled_fft))
    plt.show()










