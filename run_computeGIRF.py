# Script for estimating a scanner's gradient impulse response function from calibration data.

import numpy as np
import sigpy as sp
import sigpy.plot as pl
import matplotlib.pyplot as plt
import os
import scipy

import sys
sys.path.append('./scripts')
import mda_io as mi
import thin_slice_method as tsm

if __name__ == '__main__':
    # read in calibration data
    folder_calib = 'data/20250325/hup6/20250325_chirp_20kHz'
    # data_chirp is (n_ro, n_slice, -/+, n_avg, x/y/z, n_ch)
    data_chirp = mi.readmda(os.path.join(folder_calib, 'ADC0.mda'))
    # data_com is (n_ro, n_slice, -/+, n_avg, x/y/z, n_ch)
    data_com = mi.readmda(os.path.join(folder_calib, 'ADC2.mda'))

    # read in nominal chirp waveform
    amplitude = 1.0
    waveform_nom = amplitude * mi.readmda(os.path.join('waveforms', 'chirp_20kHz_v2.mda'))

    # set sequence parameters
    seq_params = {}
    seq_params['fov'] = 128  # mm
    seq_params['dwell_time'] = 1  # us; dwell time of readout during test waveform
    seq_params['gamma'] = 42.5764  # MHz/T
    seq_params['slice_spacing'] = 5  # mm

    # compute the GIRF
    max_iter = 30
    lamda = 0
    kernel_size = 128
    girf, waveform_meas = tsm.girf(waveform_nom, data_com, data_chirp, seq_params, kernel_size=kernel_size,
                                   lamda=lamda, max_iter=max_iter)

    # use the GIRF to predict the true waveform from the nominal waveform
    waveform_pred = tsm.predicted_waveforms(waveform_nom, girf)

    # save the GIRF
    scanner = 'hup6'
    mi.write_var(os.path.join('girfs', scanner + '.pkl'), girf)

    ## Plotting
    # plot nominal, measured, and predicted waveforms
    axis = 'y'
    times_nom = np.arange(0, len(waveform_nom)) * 10
    plt.plot(times_nom, waveform_nom, label='original')
    times_meas = np.arange(0, len(waveform_meas[axis][1]))
    plt.plot(times_meas, waveform_meas[axis][1], label='measured')
    times_pred = np.arange(0, len(waveform_pred[axis]))
    plt.plot(times_pred, waveform_pred[axis], label='predicted')
    plt.legend()

    # plot GIRFS
    for axis in ['x', 'y', 'z']:
        plt.plot(girf[axis], label=axis)
        print('Axis ' + axis + " sum:", np.sum(girf[axis]))
    plt.legend()