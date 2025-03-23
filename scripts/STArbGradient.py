# Implementation of the SequenceTree STArbGradient class in Python.
# Objects of this class are intended to mimic the behavior of SequenceTree's STArbGradient functions.

import numpy as np
import gradient_util as gu
import thin_slice_method as tsm

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
        self.fov = fov
        self.dwell_time = dwell_time
        self.kspace_offset = kspace_offset
        self.gamma = gamma
        self.girf = girf

        self._prepare()
        self._peakAmp()
        self._peakSlew()
        self._accumulatedMoment()
        self._trajectory()

    # Computes gradient amplitudes and corresponding total moments.
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
        #amp = np.where(np.logical_and(t >= 0, t < 1), (self._momentAt(t+10/self.plateau_time)-self._momentAt(t)) / 10, amp)

        # set gradient amplitudes for 2nd ramp
        holdt = 1 - (idx*10-self.ramp_time_1-self.plateau_time)/self.ramp_time_2
        holdt = np.stack(3 * [holdt], axis=0)
        amp = np.where(t >= 1, holdt*final_amplitude, amp)
        self.amp = amp

        # perform GIRF correction if one is provided
        if self.girf is not None:
            self.amp_programmed = np.copy(self.amp)
            self.amp = self._correctAmp(self.girf, self.amp_programmed)

        # compute total moments along the gradient waveform (not accounting for the preceding encoding gradient)
        total_moment = self.amp * 10
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
    # Assumes encoding gradient nulls ramp 1 moment and sets initial moment for first readout gradient point.
    def _accumulatedMoment(self):
        # encode segment nulls ramp 1 moment and sets moment for first readout gradient point
        tmp_encode_moment = -self._ramp1Moment() + self._momentAt(0)
        # the actual moment is the encode moment + the moment accumulated by the gradient waveform itself
        self.accumulated_moment = (tmp_encode_moment.T + self.total_moment.T).T

    # Computes the k-space trajectory of the arbitrary gradient from the accumulated moment.
    # Resamples the accumulated moment according to the dwell time.
    def _trajectory(self):
        N = int(self.duration() / 10) + 1
        times = np.linspace(0, self.duration(), N)

        # determine indices of readout ("plateau")
        tmp_cond = np.logical_and(times >= self.ramp_time_1, times <= self.ramp_time_1+self.plateau_time)
        idx_roGrad = np.squeeze(np.argwhere(tmp_cond))
        times = times[idx_roGrad]
        tmp_roGrad_accu_mom = self.accumulated_moment[:, idx_roGrad]

        # linear interpolation of the accumulated moments along the readout gradient
        times_interp = np.arange(times[0], times[len(times)-1], self.dwell_time)
        tmp_roGrad_accu_mom_interp = np.array([np.interp(times_interp, times, tmp_roGrad_accu_mom[i_axis]) for i_axis in range(3)])
        self.trajectory = gu.moment2kspace(tmp_roGrad_accu_mom_interp, self.fov, gamma=self.gamma)

    # Outputs total duration of STArbGradient, including both ramp times and the plateau (readout gradient) time.
    # Returns:
        # 'duration' (float): duration of STArbGradient (us)
    def duration(self):
        return self.ramp_time_1 + self.plateau_time + self.ramp_time_2

    # Outputs moments along each gradient axis as a function of time.
    # Args:
        # 't' (float or array): parameterization of the readout gradient; 0 <= t <= 1; [m1, ..., mD]
    # Returns:
        # 'moments' (float or array): gradient moments at time t; [n_dims=3, m1, ..., mD]
    def _momentAt(self, t):
        tmp_kspace_offset = self.kspace_offset
        if not (isinstance(t, float) or isinstance(t, int)):
            tmp_kspace_offset = np.reshape(tmp_kspace_offset, [3] + len(t.shape)*[1])
        return gu.kspace2moment(self._gradientShape(t) + tmp_kspace_offset, self.fov, gamma=self.gamma)

    # Outputs a desired k-space trajectory as a function of time.
    # Args:
        # 't' (float or array): parameterization of the readout gradient; 0 <= t <= 1; [m1, ..., mD]
    # Returns:
        # 'kspace' (float or array): k-space position at time t; [n_dims=3, m1, ..., mD]
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

    # # Computes a true gradient, taking a GIRF (gradient impulse response function) as argument.
    # # Also uses the true gradient to compute a corrected trajectory.
    # def correctGradientAndTrajectory(self, girf):
    #     self.girf = girf
    #     tmp_trueamp = np.copy(self.amp)
    #     # true gradient amplitude along each axis is calculated by applying the corresponding axis GIRF
    #     #TODO: implement a rotation of the FOV before GIRF correction
    #     axes = ['x', 'y', 'z']
    #     for i_axis, axis in enumerate(axes):
    #         tmp_trueamp[i_axis] = tsm.predicted_waveforms(self.amp[i_axis], girf)[axis]
    #     self.trueamp = tmp_trueamp
    #
    #     # compute total moments along the gradient waveform (not accounting for the preceding encoding gradient)
    #     total_moment = tmp_trueamp * 10
    #     total_moment = np.cumsum(total_moment, axis=1)
    #     self.total_moment = total_moment



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
        # 'kspace' (float or array): k-space position at time t; [n_dims=3, m1, ..., mD]
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