# Implementation of the SequenceTree STArbGradient class in Python.
# Objects of this class are intended to mimic the behavior of SequenceTree's STArbGradient functions.

import numpy as np
import gradient_util as gu
import thin_slice_method as tsm

# Class to model gradient calculation, trajectory estimation, and GIRF correction with a SequenceTree sequence.
# Args:
    # 'ramp_time_1' (float): first ramp time of STArbGradient (us)
    # 'plateau_time' (float): plateau time of STArbGradient (i.e. the actual arbitrary gradient); (us)
    # 'ramp_time_2' (float): second ramp time of STArbGradient (us)
    # 'fov' (array): field-of-view of acquisition [x/y/z] (mm)
    # 'dwell_time' (float): dwell time of readout
    # 'kspace_offset' (array): offset in k-space applied to all points in the trajectory; [x,y,z]
    # 'gamma' (float): gyromagnetic ratio (MHz/T)
    # 'girf' (dict): gradient impulse response function for trajectory correction, sampled on 1 us increment
class STArbGradient(object):
    def __init__(
            self,
            ramp_time_1,
            plateau_time,
            ramp_time_2,
            fov,
            dwell_time,
            kspace_offset=np.array([0,0,0]),
            gamma=42.5764,
            girf=None,
    ):
        self.ramp_time_1 = ramp_time_1
        self.plateau_time = plateau_time
        self.ramp_time_2 = ramp_time_2
        self.fov = np.array(fov)
        self.dwell_time = dwell_time
        self.kspace_offset = kspace_offset
        self.gamma = gamma
        self.girf = girf

        self._prepare()
        self._peakAmp()
        self._peakSlew()
        self._accumulatedMoment()
        self._trajectory()

    # Computes programmed gradient amplitudes and corresponding total moments.
    # Calculates corrected gradients if GIRF exists.
    def _prepare(self):
        # number of time points for discretizing the gradient waveform, including ramps
        N = int(self.duration()/10) + 1

        # initial and final amplitudes of readout gradient estimated by numerical derivatives
        initial_amplitude = (self._momentAt(10/self.plateau_time) - self._momentAt(0)) / 10
        final_amplitude = (self._momentAt(1) - self._momentAt(1-10/self.plateau_time)) / 10
        initial_amplitude = np.reshape(initial_amplitude, [3, 1])
        final_amplitude = np.reshape(final_amplitude, [3, 1])


        # create discrete time points for calculating gradient amplitudes
        idx = np.array([j for j in range(N)])
        t = np.array([(j*10-self.ramp_time_1)/self.plateau_time for j in range(N)])

        # initialize gradient amplitudes
        amp = np.zeros((3, N))
        # set gradient amplitudes for 1st ramp
        holdt = idx*10/self.ramp_time_1
        holdt = np.stack(3*[holdt], axis=0)
        amp = np.where(t < 0, holdt*initial_amplitude, amp)

        # set gradient amplitudes during readout gradient waveform
        idx_gradWav = np.squeeze(np.argwhere(np.logical_and(t >= 0, t < 1)))
        amp[:, idx_gradWav] = (self._momentAt(t[idx_gradWav]+10/self.plateau_time)-self._momentAt(t[idx_gradWav])) / 10

        # set gradient amplitudes for 2nd ramp
        holdt = 1 - (idx*10-self.ramp_time_1-self.plateau_time)/self.ramp_time_2
        holdt = np.stack(3 * [holdt], axis=0)
        amp = np.where(t >= 1, holdt*final_amplitude, amp)
        # amp is programmed on 10 us increment
        self.amp = amp

        # interpolate amp onto 1 us increment
        times_10us = t*self.plateau_time
        times_1us = np.arange(times_10us[0], times_10us[len(times_10us)-1]+1)
        amp_1us = np.zeros((3, len(times_1us)))
        for i_axis in range(3):
            amp_1us[i_axis] = np.interp(times_1us, times_10us, self.amp[i_axis])
        self.amp_1us = amp_1us

        # perform GIRF correction on the 1 us amp if one is provided
        if self.girf is not None:
            self.amp_1us = self._correctAmp(self.girf, self.amp_1us)

        # compute total moments along the gradient waveform (not accounting for the preceding encoding gradient)
        total_moment = self.amp_1us * 1
        total_moment = np.cumsum(total_moment, axis=1)
        self.total_moment = total_moment

    # Computes the peak gradient amplitude along the waveform.
    def _peakAmp(self):
        self.peakamp = np.max(np.abs(self.amp))

    # Computes the peak gradient slew along the waveform.
    def _peakSlew(self):
        N = int(self.duration() / 10) + 1
        tmp_amp_tp10 = self.amp[:, 1:]
        tmp_amp_t = self.amp[:, :N-1]
        slew = (tmp_amp_tp10 - tmp_amp_t)/10
        self.peakslew = np.max(np.abs(slew))

    # Computes the accumulated moment of the arbitrary gradient along with the encoding gradient.
    # Assumes encoding gradient nulls ramp 1 moment and sets up initial moment for first readout gradient point.
    def _accumulatedMoment(self):
        # encode segment nulls ramp 1 moment and sets moment for first readout gradient point
        tmp_encode_moment = -self._ramp1Moment() + self._momentAt(0)
        # the actual moment is the encode moment + the moment accumulated by the gradient waveform itself
        self.accumulated_moment = (tmp_encode_moment.T + self.total_moment.T).T

    # Computes the k-space trajectory of the arbitrary gradient from the accumulated moment.
    # Resamples the accumulated moment according to the dwell time.
    def _trajectory(self):
        # get times of arbitrary gradient, including ramps
        times = np.arange(0, self.duration()+1) - self.ramp_time_1
        # determine indices of readout "plateau"
        tmp_cond = np.logical_and(times >= 0, times < self.plateau_time)
        idx_roGrad = np.squeeze(np.argwhere(tmp_cond))
        tmp_roGrad_accum_mom = self.accumulated_moment[:, idx_roGrad]
        # sample readout, incrementing by the dwell time
        tmp_roGrad_accum_mom = tmp_roGrad_accum_mom[:, ::self.dwell_time]
        self.trajectory = gu.moment2kspace(tmp_roGrad_accum_mom, self.fov, gamma=self.gamma)

    # Outputs total duration of STArbGradient, including both ramp times and the plateau (readout gradient) time.
    # Returns:
        # 'duration' (float): duration of STArbGradient (us)
    def duration(self):
        return self.ramp_time_1 + self.plateau_time + self.ramp_time_2

    # Outputs moments along each gradient axis as a function of time.
    # Args:
        # 't' (float or array): parameterization of the readout gradient; 0 <= t <= 1; [m1, ..., mD]
    # Returns:
        # 'moments' (float or array): gradient moments at time t; [x/y/z, m1, ..., mD]
    def _momentAt(self, t):
        tmp_kspace_offset = self.kspace_offset
        if not (isinstance(t, float) or isinstance(t, int)):
            tmp_kspace_offset = np.reshape(tmp_kspace_offset, [3] + len(t.shape)*[1])
        return gu.kspace2moment(self._gradientShape(t) + tmp_kspace_offset, self.fov, gamma=self.gamma)

    # Outputs a desired k-space trajectory as a function of time.
    # Args:
        # 't' (float or array): parameterization of the readout gradient; 0 <= t <= 1; [m1, ..., mD]
    # Returns:
        # 'kspace' (float or array): k-space position at time t; [x/y/z, m1, ..., mD]
    def _gradientShape(self, t):
        return

    # Returns moment of the first ramp.
    def _ramp1Moment(self):
        initial_amplitude = (self._momentAt(10/self.plateau_time) - self._momentAt(0)) / 10
        return initial_amplitude * self.ramp_time_1 / 2

    # Computes the GIRF-corrected gradient amplitudes given a set of programmed gradient amplitudes.
    def _correctAmp(self, girf, amp):
        newamp = np.copy(amp)
        # true gradient amplitude along each axis is calculated by applying the corresponding axis GIRF
        # TODO: implement a rotation of the FOV before GIRF correction
        axes = ['x', 'y', 'z']
        for i_axis, axis in enumerate(axes):
            newamp[i_axis] = tsm.predicted_waveforms(self.amp[i_axis], girf)[axis]
        return newamp


# Example implementation of class that inherits from STArbGradient.
# Identical implementation of STCircleGradient in SequenceTree.
class STCircleGradient(STArbGradient):
    def __init__(
            self,
            kspace_radius_1,
            kspace_radius_2,
            num_cycles,
            kspace_direction_1,
            kspace_direction_2,
            ramp_time_1,
            plateau_time,
            ramp_time_2,
            fov,
            dwell_time,
            kspace_offset=np.array([0, 0, 0]),
            gamma=42.5764,
    ):
        self.kspace_radius_1 = kspace_radius_1
        self.kspace_radius_2 = kspace_radius_2
        self.num_cycles = num_cycles
        self.kspace_direction_1 = kspace_direction_1
        self.kspace_direction_2 = kspace_direction_2
        super().__init__(ramp_time_1, plateau_time, ramp_time_2, fov, dwell_time, kspace_offset=kspace_offset, gamma=gamma)


    # Outputs a desired k-space trajectory as a function of time.
    # Args:
        # 't' (float or array): parameterization of the readout gradient; 0 <= t <= 1; [m1, ..., mD]
    # Returns:
        # 'kspace' (float or array): k-space position at time t; [x/y/z, m1, ..., mD]
    def _gradientShape(self, t):
        t2 = t * 2 * 3.141592*self.num_cycles
        tmp_kspace_direction_1 = self.kspace_direction_1
        tmp_kspace_direction_2 = self.kspace_direction_2
        if not (isinstance(t, float) or isinstance(t, int)):
            tmp_kspace_direction_1 = np.reshape(tmp_kspace_direction_1, [3] + len(t.shape)*[1])
            tmp_kspace_direction_2 = np.reshape(tmp_kspace_direction_2, [3] + len(t.shape)*[1])
        tmp_x = tmp_kspace_direction_1*self.kspace_radius_1*np.cos(t2)
        tmp_y = tmp_kspace_direction_2*self.kspace_radius_2*np.sin(t2)
        return tmp_x + tmp_y