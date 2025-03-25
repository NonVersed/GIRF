# Functions for computation on gradients, k-space, and moments.
# Default units and conventions are those used by SequenceTree.
# Functions mimic SequenceTree behavior as much as possible.

import numpy as np


# Computes moments based on k-space location.
# Args:
    # 'kspace' (array): coordinates in kspace; [x/y/z, m1, ..., mD]
    # 'fov' (array): field-of-view along x/y/z; 1D array of length 3 [x/y/z] (mm)
    # 'gamma' (float): gyromagnetic ratio (MHz/T)
# Returns:
    # (array) moments along each axis; [x/y/z, m1, ..., mD]
def kspace2moment(kspace, fov, gamma=42.5764):
    tmp_fov = np.reshape(fov, [3] + (len(kspace.shape)-1)*[1])
    return kspace/(tmp_fov * gamma)*1E6


# Computes k-space location based on moments.
# Args:
    # 'moment' (array): moments along each axis; [x/y/z, m1, ..., mD]; [uT/mm]-us
    # 'fov' (array): field-of-view along x/y/z; 1D array of length 3 [x/y/z] (mm)
    # 'gamma' (float): gyromagnetic ratio (MHz/T)
# Returns:
    # (array) k-space coordinates; [x/y/z, m1, ..., mD]
def moment2kspace(moment, fov, gamma=42.5764):
    tmp_fov = np.reshape(fov, [3] + (len(moment.shape) - 1) * [1])
    return moment*(tmp_fov * gamma)/1E6



# # Computes a gradient waveform based on an array of moments, assuming the moments are spaced 10 us apart.
# # NOTE: This implementation is identical to how gradients are computed in SequenceTree!
# # Args:
#     # 'moment_10': array of moments along each axis; [x/y/z, n_steps+1, m1, ..., mD]; [uT/mm]-us
#         # 'n_steps': number of 10 us intervals in 'plateau_time', the duration of the arbitrary gradient
# # Returns:
#     # array of gradient amplitudes; [x/y/z, n_steps, m1, ..., mD]; uT/mm
# def moment2gradient(moment_10):
#     n_steps = moment_10.shape[1]-1
#     moment_10_plus10 = np.roll(moment_10, -1, axis=1)
#     gradient_10 = (moment_10_plus10 - moment_10)/10
#     return gradient_10[:, :n_steps]
#
#
# # Computes accumulated moment based on a gradient waveform, assuming the gradient values are spaced 10 us apart.
# # NOTE: This implementation is identical to how moments are computed in SequenceTree!
# # Args:
#     # 'gradient_10': array of gradients along each axis; [x/y/z, n_steps, m1, ..., mD]; [uT/mm]-us
#         # 'n_steps': number of 10 us intervals in 'plateau_time', the duration of the arbitrary gradient
# # Returns:
#     # array of moments; [x/y/z, n_steps, m1, ..., mD]; uT/mm
# def gradient2moment(gradient_10):
#     moment_10 = np.cumsum(gradient_10*10, axis=1)
#     return moment_10