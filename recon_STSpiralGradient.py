# Script to reconstruct images acquired by STSpiralGradient (with optional GIRF correction).

import numpy as np
import sys
import sigpy as sp
import sigpy.plot as pl
import sigpy.mri as mr
import os
import nibabel as nib

sys.path.append('./scripts')
from STSpiralGradient import STSpiralGradient
import mda_io as mi

if __name__ == '__main__':
    # specify sequence parameters relating to the trajectory
    n_shots = 48
    shot_angles = [2*np.pi/n_shots*i for i in range(n_shots)] # rad
    fov = 256  # mm
    slice_thickness = 2 # mm
    matrix_size = 256
    gamma = 42.5764  # MHz/T
    maxamp = 25  # uT/mm
    ramprate = 0.1  # uT/mm/us
    dwell_time = 2  # us
    ramp_time_1 = 100 # us
    ramp_time_2 = 200 # us


    ## READ RAW DATA
    folder_ksp = 'data/20250325/hup6/spiral_isocenter'
    ksp = mi.readmda(os.path.join(folder_ksp, 'ADC1.mda')).T
    if len(ksp.shape) == 2:
        n_ch = 1
        ksp = np.reshape(ksp, [1] + list(ksp.shape))
    else:
        n_ch = ksp.shape[0]

    # get the scanner girf
    girf = mi.read_var('girfs/hup6.pkl')


    ## NOMINAL TRAJECTORY
    # generate spiral gradients for each shot angle and append them to the list
    spiral_gradients = []
    for shot_angle in shot_angles:
        tmp_grad = STSpiralGradient(shot_angle, n_shots, matrix_size, maxamp, ramprate, ramp_time_1, ramp_time_2, fov,
                                    slice_thickness, dwell_time, gamma=gamma)
        spiral_gradients.append(tmp_grad)
        del tmp_grad

    # nominal k-space trajectory
    trajectory = np.stack([spiral_gradients[i].trajectory for i in range(len(spiral_gradients))], axis=-1).T
    trajectory = trajectory[:, :, 0:2]

    # compute DCF (nominal trajectory)
    dcf = mr.pipe_menon_dcf(trajectory)
    img_shape = 2 * [fov]
    F = sp.linop.NUFFT([n_ch] + img_shape, trajectory)
    # coil-by-coil NUFFT adjoint
    img_coil = F.H * (dcf * ksp)
    # RSS coil combination
    img_rss = sp.rss(img_coil, axes=(0,))


    ## CORRECTED TRAJECTORY
    # generate spiral gradients for each shot angle and append them to the list
    spiral_gradients_corrected = []
    for shot_angle in shot_angles:
        tmp_grad = STSpiralGradient(shot_angle, n_shots, matrix_size, maxamp, ramprate, ramp_time_1, ramp_time_2, fov,
                                    slice_thickness, dwell_time, gamma=gamma, girf=girf)
        spiral_gradients_corrected.append(tmp_grad)
        del tmp_grad

    # corrected k-space trajectory, according to GIRF measurement
    trajectory_girf = np.stack(
        [spiral_gradients_corrected[i].trajectory for i in range(len(spiral_gradients_corrected))], axis=-1).T
    trajectory_girf = trajectory_girf[:, :, 0:2]

    # compute DCF (GIRF-corrected trajectory)
    dcf_girf = mr.pipe_menon_dcf(trajectory_girf)
    img_shape = 2 * [fov]
    F = sp.linop.NUFFT([n_ch] + img_shape, trajectory_girf)
    # coil-by-coil NUFFT adjoint
    img_coil_girf = F.H * (dcf_girf * ksp)
    # RSS coil combination
    img_rss_girf = sp.rss(img_coil_girf, axes=(0,))


    ## PLOTTING
    pl.ImagePlot(np.array([img_rss, img_rss_girf]))

    ## SAVE AS NIFTI
    nifti_rss = nib.Nifti1Image(img_rss, np.eye(4))
    nib.save(nifti_rss, os.path.join(folder_ksp, 'rss.nii'))
    nifti_rss_girf = nib.Nifti1Image(img_rss_girf, np.eye(4))
    nib.save(nifti_rss_girf, os.path.join(folder_ksp, 'rss_girf.nii'))