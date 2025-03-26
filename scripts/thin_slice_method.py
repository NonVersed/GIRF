# Functions implementing the off-isocenter thin slice method and gradient impulse response function (GIRF).

import numpy as np
import sigpy as sp


# Computes the centers of mass for a series of slices.
# Args:
    # 'com_meas' (array): array of measurements from center-of-mass measurement sequence; [n_ro, -/+, n_slices, m1, ..., mD]
    # 'fov' (float): length of field-of-view (mm); scalar
    # 'slice_spacing' (float): spacing between slices; scalar in mm
    # 'threshold' (float): value between 0 and 1 for thresholding signal and noise from slice-selection; scalar
# Returns:
    # (array) locations of the centers of mass; [n_slices, m1, ..., mD]
def center_of_mass(com_meas, fov, slice_spacing, threshold=0.05):
    data_com = np.copy(com_meas)
    # get number of slices
    n_slices = data_com.shape[2]
    # compute nominal slice positions
    nom_slice_pos = np.arange(-n_slices // 2, -n_slices // 2 + n_slices, 1) * slice_spacing

    # flip readouts going in the negative direction
    data_com[:, 0] = np.flip(data_com[:, 0], axis=0)
    # convert k-space to slice profiles
    img_com = sp.ifft(data_com, axes=(0,))

    # threshold readout data
    peak_values = np.max(np.abs(img_com), axis=0)
    img_com_thresh = (np.abs(img_com) > threshold*peak_values) * img_com

    ## estimate the centers of mass
    n_ro = com_meas.shape[0]

    # TODO: positions are assumed to be centric; could be left-sided or right-sided
    #positions = np.linspace(-(fov / 2 - fov / n_ro / 2), fov / 2 - fov / n_ro / 2, n_ro)
    # TODO: positions are probably left sided because we use the FFT
    positions = np.arange(-fov/2, fov/2, fov/n_ro)

    # convert the thresholded slice profiles into weights
    img_weights = np.abs(img_com_thresh) / np.sum(np.abs(img_com_thresh), axis=0)
    # get the mean displacement, weighted by the signal at each position
    mean_disp = np.sum(positions * np.moveaxis(img_weights, 0, -1), axis=-1)
    # average the positive and negative displacements
    com_disp = np.mean(mean_disp, axis=0)

    # add the center-of-mass displacements to the nominal slice positions
    slice_positions = np.reshape(nom_slice_pos, [n_slices] + (len(com_disp.shape) - 1) * [1]) + com_disp

    return slice_positions


# Computes the difference of angular derivatives (time evolution of local magnetic field).
# This function implements Eqn (4.17) (which is slightly incorrect) of Paul Gurney's PhD thesis at Stanford.
# Args:
    # 'grad_meas' (array): array of measurements from gradient waveform measurement sequence; [n_ro, -/+, m1, ..., mD]
    # 'dwell_time' (float): spacing between sampled points in time; scalar in us
    # 'gamma' (float): gyromagnetic ratio; scalar in MHz/T
# Returns:
    # (array) time evolution of local magnetic field; [n_ro, m1, ..., mD]; values in uT
def time_evol_field(grad_meas, dwell_time, gamma=42.5764):
    data_grad = grad_meas
    n_ro = data_grad.shape[0]

    # compute right-sided derivative
    tmp_upper = np.roll(data_grad, -1, axis=0)
    tmp_lower = data_grad
    ang_deriv_rhs = ( tmp_upper * np.conj(tmp_lower) )**( 1/(2*np.pi*gamma*dwell_time) )
    # compute difference of angular derivatives
    diff_rhs = ( ang_deriv_rhs[:, 1] * np.conj(ang_deriv_rhs[:, 0]) )**(1/2)

    # compute left-sided derivative
    tmp_upper = data_grad
    tmp_lower = np.roll(data_grad, 1, axis=0)
    ang_deriv_lhs = ( tmp_upper * np.conj(tmp_lower) )**( 1/(2*np.pi*gamma*dwell_time) )
    # compute difference of angualr derivatives
    diff_lhs = ( ang_deriv_lhs[:, 1] * np.conj(ang_deriv_lhs[:, 0]) )**(1/2)

    # get phase of rhs and lhs computations
    time_evol_rhs = np.angle(diff_rhs) * 1E6
    time_evol_lhs = np.angle(diff_lhs) * 1E6

    # take average of left- and right-handed estimates of the phase
    time_evol = np.zeros_like(time_evol_rhs)
    # first point uses rhs derivative only
    time_evol[0] = time_evol_rhs[0]
    # last point uses lhs derivative only
    time_evol[n_ro-1] = time_evol_lhs[n_ro-1]
    # all other points are the average of the rhs and lhs derivatives
    time_evol[1:n_ro-1] = 0.5 * (time_evol_lhs[1:n_ro-1] + time_evol_rhs[1:n_ro-1])

    return time_evol


# Computes the weights for least-squares estimation of the B0 and gradient waveforms.
# Samples are weighted by their corresponding signal magnitude.
# Args:
    # 'grad_meas' (array): measurements from gradient waveform measurement sequence; [n_ro, -/+, m1, ..., mD]
# Returns:
    # (array) sample weights; [n_ro, m1, ..., mD]
def sample_weights(grad_meas):
    data_grad = np.copy(grad_meas)
    n_ro = data_grad.shape[0]
    # shift gradient data
    data_grad_shift = np.roll(data_grad, -1, axis=0)
    data_grad_shift[n_ro - 1] = data_grad_shift[n_ro - 2]

    weights = np.sqrt( np.abs(data_grad[:,1]*data_grad_shift[:,1] * data_grad[:,0]*data_grad_shift[:,0]) )
    return weights


# Computes the B0 and gradient waveforms from the measurements.
# Args:
    # 'com_meas' (array): measurements from center-of-mass measurement sequence; [n_ro_com, n_slices, -/+, n_avgs, x/y/z, n_ch]
    # 'grad_meas' (array): measurements from gradient waveform measurement sequence; [n_ro_grad, n_slices, -/+, n_avgs, x/y/z, n_ch]
    # 'seq_params' (dict): pulse sequence parameters, keys listed below
        # 'dwell_time' (float): spacing between sampled points in time (us)
        # 'fov' (float): length of field-of-view (mm)
        # 'slice_spacing (float)': spacing between slices (mm)
        # 'gamma' (float): gyromagnetic ratio (MHz/T)
    # 'threshold' (float): value between 0 and 1 for thresholding signal and noise from slice-selection
# Returns:
    # (dict) B0 and gradient waveform measurements for each axis (x/y/z)
def measured_waveforms(com_meas, grad_meas, seq_params, threshold=0.05):
    dwell_time = seq_params['dwell_time']
    fov = seq_params['fov']
    slice_spacing = seq_params['slice_spacing']
    gamma = seq_params['gamma']

    data_grad = np.copy(grad_meas)
    data_com = np.copy(com_meas)

    # compute the slice positions (in mm) from the center-of-mass measurements
    data_com = np.moveaxis(data_com, 2, 1)
    slice_positions = center_of_mass(data_com, fov, slice_spacing, threshold=threshold)

    # compute the time evolution of the magnetic fields (in uT)
    data_grad = np.moveaxis(data_grad, 2, 1)
    field_evol = time_evol_field(data_grad, dwell_time, gamma)

    # reshape arrays to prepare for least squares calculation
    slice_positions_flat = np.moveaxis(slice_positions, -2, 0)
    slice_positions_flat = np.reshape(slice_positions_flat, [slice_positions_flat.shape[0], -1])
    del slice_positions

    n_ro_chirp = data_grad.shape[0]
    field_evol_flat = np.moveaxis(field_evol, -2, 0)
    field_evol_flat = np.reshape(field_evol_flat, [field_evol_flat.shape[0], n_ro_chirp, -1])
    del field_evol

    # num of rows of A matrix
    n_samples = slice_positions_flat.shape[1]

    # create weights matrix
    weights = sample_weights(data_grad)
    weights_flat = np.moveaxis(weights, -2, 0)
    weights_flat = np.reshape(weights_flat, list(weights_flat.shape[0:2]) + [-1])
    del weights

    del data_com, data_grad

    # loop over each axis to compute the measured waveform
    measured_gradient_waveforms = []
    for i_axis in range(3):
        # get weights matrix for this axis
        W = weights_flat[i_axis]
        # create A matrix
        A = np.stack([np.ones(n_samples), slice_positions_flat[i_axis]], axis=1)
        # get field measurements
        y = field_evol_flat[i_axis]
        # weighted least-squares
        x = np.array(
            [np.linalg.lstsq(np.reshape(W[i_ro], [n_samples, 1]) * A, W[i_ro] * y[i_ro], rcond=-1)[0] for i_ro in
             range(n_ro_chirp)]).T
        measured_gradient_waveforms.append(x)
        del W, A, y, x

    return {'x': measured_gradient_waveforms[0],
            'y': measured_gradient_waveforms[1],
            'z': measured_gradient_waveforms[2]}


# Computes the gradent impulse response function from calibration data.
# Args:
    # 'waveform_nom' (array): nominal gradient waveform; [duration/10 + 1]; programmed in increments of 10 us
    # 'com_meas' (array): measurements from center-of-mass measurement sequence; [n_ro_com, n_slices, -/+, n_avgs, x/y/z, n_ch]
    # 'grad_meas' (array): measurements from gradient waveform measurement sequence; [n_ro_grad, n_slices, -/+, n_TEs, n_avgs, x/y/z, n_ch]
    # 'seq_params' (dict): pulse sequence parameters, keys listed below
        # 'dwell_time' (float): spacing between sampled points in time (us)
        # 'fov' (float): length of field-of-view (mm)
        # 'slice_spacing' (float): spacing between slices (mm)
        # 'gamma' (float): gyromagnetic ratio (MHz/T)
    # 'threshold' (float): value between 0 and 1 for thresholding signal and noise from slice-selection
    # 'kernel_size' (int): duration of GIRF kernel; increment of 1 us
    # 'lamda' (float): regularization parameter for calculating GIRF
    # 'max_iter' (int): number of CG iterations for estimating GIRF
# Returns:
    # (dict) gradient impulse response functions (i.e. convolution kernels) for each axis (x/y/z)
    # (dict) gradient waveform measurements for each axis (x/y/z)
def girf(waveform_nom, com_meas, grad_meas, seq_params, threshold=0.05, kernel_size=320, lamda=0, max_iter=100):
    # compute the measured B0 and gradient waveforms
    wav_meas = measured_waveforms(com_meas, grad_meas, seq_params, threshold=threshold)

    # interpolate the nominal waveform (programmed in 10 us) to 1 us increment
    times_10us = np.arange(0, len(waveform_nom)) * 10
    times_1us = np.arange(0, len(grad_meas))*seq_params['dwell_time']
    wav_nom = np.interp(times_1us, times_10us, waveform_nom)
    n_pts = len(wav_nom)

    # estimate the GIRF for each axis, store in dictionary
    girf = {}
    for axis in ['x', 'y', 'z']:
        tmp_W_nom = sp.linop.ConvolveFilter([kernel_size], wav_nom)
        tmp_R = sp.linop.Resize([n_pts], tmp_W_nom.oshape)
        tmp_app = sp.app.LinearLeastSquares(tmp_R * tmp_W_nom, wav_meas[axis][1], lamda=lamda, max_iter=max_iter)
        tmp_ker = tmp_app.run()
        girf[axis] = tmp_ker
        del tmp_W_nom, tmp_R, tmp_app, tmp_ker

    return girf, wav_meas


# Predicts the true gradient waveforms along each axis using the gradient impulse response function.
# Args:
    # 'waveform_nom' (array): nominal gradient waveform; [duration/10 + 1]; programmed in increments of 10 us
    # 'girf' (dict): gradient impulse response functions (i.e. convolution kernels) for each axis (x/y/z)
# Returns:
    # (dict) predicted waveforms for each axis (x/y/z); increments of 1 us
def predicted_waveforms(waveform_nom, girf):
    # interpolate the nominal waveform (programmed in 10 us) to 1 us increment
    times_10us = np.arange(0, len(waveform_nom)) * 10
    times_1us = np.arange(0, 10*(len(waveform_nom)-1)+1)
    waveform_nom_1us = np.interp(times_1us, times_10us, waveform_nom)

    # check if nominal waveform is odd in length
    isOdd = waveform_nom_1us.shape[0]%2 == 1
    # pad with 0 at end to make length even
    if isOdd:
        wav_nom = np.pad(waveform_nom_1us, (0, 1), mode='constant')

    n_ro = wav_nom.shape[0]

    waveform_pred = {}
    for axis in ['x', 'y', 'z']:
        tmp_K = sp.linop.ConvolveData([n_ro], girf[axis])
        tmp_R = sp.linop.Resize([n_ro], tmp_K.oshape)
        tmp_wav_pred = tmp_R * tmp_K * wav_nom

        # take off the padded index if the nominal waveform was odd in length
        if isOdd:
            tmp_wav_pred = tmp_wav_pred[:n_ro-1]

        waveform_pred[axis] = tmp_wav_pred
        del tmp_K, tmp_R, tmp_wav_pred

    return waveform_pred