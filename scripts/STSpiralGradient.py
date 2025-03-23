# Class implementing an Archemedean spiral readout in 2D.
# Inherits from the STArbGradient class, which mimics SequenceTree behavior.
# Based on "Bernstein, Matt A., Kevin F. King, and Xiaohong Joe Zhou. Handbook of MRI pulse sequences. Elsevier, 2004."
# Section 17.6

import numpy as np
from STArbGradient import STArbGradient

class STSpiralGradient(STArbGradient):
    def __init__(
            self,
            shot_angle,
            n_shots,
            matrix_size,
            maxamp,
            ramprate,
            ramp_time_1,
            ramp_time_2,
            fov_xy,
            slice_thickness,
            dwell_time,
            kspace_offset=np.array([0, 0, 0]),
            gamma=42.5764,
            girf=None,
    ):
        self.shot_angle = shot_angle
        self.n_shots = n_shots
        self.matrix_size = matrix_size
        self.maxamp = maxamp
        self.ramprate = ramprate
        self.fov_xy = fov_xy
        fov = 2*[fov_xy] + [slice_thickness] # z dim is a dummy parameter, spiral is 2D
        self.gamma = gamma
        self.dwell_time = dwell_time

        # compute intermediate parameters
        self._intermediateParams()

        super().__init__(ramp_time_1, self.plateau_time, ramp_time_2, fov, dwell_time, kspace_offset=kspace_offset,
                         gamma=gamma, girf=girf)

    def _intermediateParams(self):
        lamda = self.n_shots/(2*np.pi*self.fov_xy/10)
        beta = self.gamma*1E2 * self.ramprate*1E5 / lamda
        a2 = (9*beta/4) ** (1/3)
        # Ts is the transition time between slew rate limited segment and the gradient amp limited segment
        Ts = ((3 * self.gamma*1E2 * self.maxamp/10) / (2*lamda*a2**2))**3
        # theta_s is the transition angle between the slew rate limited segment adn the gradient amp limited segment
        theta_s = (0.5 * beta * Ts**2) / (1 + beta/(2*a2) * Ts**(4/3))
        # theta_max is the final angle of the spiral readout
        theta_max = np.pi*self.matrix_size/self.n_shots

        # set class params
        self.lamda = lamda
        self.beta = beta
        self.a2 = a2
        self.Ts = Ts
        self.theta_s = theta_s
        self.theta_max = theta_max

        # determine if gradient is slew limited or gradient amp limited
        if theta_s < theta_max:
            self.isGradAmpLimited = True
            Tacq = Ts + lamda / (2 * self.gamma*1E2 * self.maxamp/10) * (theta_max**2 - theta_s**2)
        else:
            self.isGradAmpLimited = False
            Tacq = (2*np.pi * self.fov/10) / (3 * self.n_shots) \
                   * (2 * self.gamma*1E2 * self.ramprate*1E5 * (self.fov/10/self.matrix_size)**3)**(-1/2)
        self.Tacq = Tacq

        # determine the readout and readout gradient durations and number of points in readout
        readout_duration = Tacq * 1E6
        N = int(readout_duration / self.dwell_time)
        if N % 2 == 1:
            N = N + 1
        readout_duration = self.dwell_time * N
        self.readout_duration = readout_duration
        self.plateau_time = readout_duration

    def _gradientShape(self, t):
        time = t * self.plateau_time / 1E6

        # if argument is a float or int
        if (isinstance(t, float) or isinstance(t, int)):
            if time < self.Ts:
                theta = self._theta1(t)
            else:
                theta = self._theta2(t)
        # if argument is an array
        else:
            idx_isSlewing = np.squeeze(np.argwhere(time < self.Ts))
            idx_isNotSlewing = np.squeeze(np.argwhere(time >= self.Ts))
            theta = np.zeros_like(time)
            theta[idx_isSlewing] = self._theta1(t[idx_isSlewing])
            theta[idx_isNotSlewing] = self._theta2(t[idx_isNotSlewing])

        k = self.lamda/10 * theta * self.matrix_size
        dirX = np.array([1,0,0])
        dirY = np.array([0,1,0])
        # if arguemnt is an array
        if not (isinstance(t, float) or isinstance(t, int)):
            N = len(t)
            dirX = np.stack(N*[dirX], axis=1)
            dirY = np.stack(N*[dirY], axis=1)

        return dirX * k * np.cos(theta + self.shot_angle) + dirY * k * np.sin(theta + self.shot_angle)

    def _theta1(self, t):
        time = t * self.plateau_time / 1E6
        # fractional power casts output to complex dtype
        output = (0.5 * self.beta * time**2) / (1 + self.beta/(2*self.a2) * time**(4.0/3))
        return output
        #return np.real(output)

    def _theta2(self, t):
        time = t * self.plateau_time / 1E6
        gamma = self.gamma*1E2
        maxamp = self.maxamp/10
        # fractional power casts output to complex dtype
        output = (self.theta_s**2 + 2*gamma/self.lamda*maxamp*(time - self.Ts))**(0.5)
        return output
        #return np.real(output)