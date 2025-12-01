from abc import ABC, abstractmethod
import numpy as np

from Simulator.trackSim import Simulator
from Simulator.trackSim import SE2
from Simulator.trackSim import Indicies

class SOT(ABC):
    total_states = 6
    total_measurements = 2

    """ Parent Single Object tracking class use 
    for generic tracking functionbality """
    def __init__(self, simulator : Simulator):

        self.simulator : Simulator = simulator
        self.states = np.zeros((self.total_states, len(self.simulator.measurements)), 
                               dtype=np.float64)

        self.covariances = np.zeros((self.simulator.datapoints, 
                                     self.total_states, self.total_states), 
                                    dtype=np.float64)

        self.sample_rate = simulator.sample_rate

        self.choosen_measurements = []

        self.Q = np.zeros((self.total_states, self.total_states))
        self.R = np.zeros((self.total_measurements, self.total_measurements))

        self.set_prior()

    def set_prior(self):
        if (self.simulator.total_objects > 1):
            raise ValueError("Single object trackers can only track one object.")

        first_id : int = 0

        self.states[SE2.X.value, 0] = self.simulator.objects[first_id][SE2.X.value][0]
        self.states[SE2.Y.value, 0] = self.simulator.objects[first_id][SE2.Y.value][0]
        self.states[SE2.O.value, 0] = self.simulator.objects[first_id][SE2.O.value][0]
        self.states[SE2.V.value, 0] = self.simulator.objects[first_id][SE2.V.value][0]
        self.states[SE2.A.value, 0] = self.simulator.objects[first_id][SE2.A.value][0]
        self.states[SE2.W.value, 0] = self.simulator.objects[first_id][SE2.W.value][0]

        np.fill_diagonal(self.covariances[0], 0.01)

        np.fill_diagonal(self.Q, 0.01)
        np.fill_diagonal(self.R, self.simulator.noise)

    def prediction(self, prior, prior_cov, posterior, posterior_cov):
        posterior[SE2.X.value] =  prior[SE2.X.value] + \
            self.sample_rate * prior[SE2.V.value] * np.cos(prior[SE2.O.value])

        posterior[SE2.Y.value] = prior[SE2.Y.value] + \
            self.sample_rate * prior[SE2.V.value] * np.sin(prior[SE2.O.value])

        posterior[SE2.O.value] = prior[SE2.O.value]
        posterior[SE2.V.value] = prior[SE2.V.value]
        posterior[SE2.A.value] = .0
        posterior[SE2.W.value] = .0

        F = self.motion_jacobian(prior)
        posterior_cov[:] = F @ prior_cov @ F.T + self.Q

    @abstractmethod
    def correction(self, state, covariance, measurement_set):
        pass

    def track(self):
        for k in range(1, len(self.simulator.measurements)):
            self.prediction(self.states[ : ,k-1], 
                            self.covariances[k-1], 
                            self.states[ : , k], 
                            self.covariances[k])

            self.correction(self.states[ : , k], self.covariances[k],
                            self.simulator.measurements[k])

    def motion_jacobian(self, state : np.ndarray) -> np.ndarray: 
        jacobian = np.zeros((self.total_states, self.total_states))

        theta = state[SE2.O.value]
        v = state[SE2.V.value]

        delta_t = self.sample_rate

        jacobian[0] = [1,0, -v * np.sin(theta) * delta_t, np.cos(theta) * delta_t,0,0]
        jacobian[1] = [0,1,  v * np.cos(theta) * delta_t, np.sin(theta) * delta_t,0,0]
        jacobian[2] = [0,0,1,0,0,0]
        jacobian[3] = [0,0,0,1,0,0]

        return jacobian

    def meas_jacobian(self, state : np.ndarray) -> np.ndarray :
        jacobian = np.zeros((self.total_measurements, self.total_states))
        x = state[SE2.X.value]
        y = state[SE2.Y.value]
        r = np.sqrt(np.pow(x,2) + np.pow(y,2))

        jacobian[0] = [x/r, y/r, 0, 0, 0, 0]
        jacobian[1] = [-y/np.pow(r,2), x/np.pow(r,2), 0, 0, 0, 0]

        return jacobian

    def predicted_meas(self, state) -> np.ndarray:
        predicted_meas = np.zeros(2, dtype=np.float64)

        predicted_meas[Indicies.RANGE.value] =\
            np.sqrt(np.pow(state[SE2.X.value],2) +\
                np.pow(state[SE2.Y.value],2))

        predicted_meas[Indicies.BEARING.value] =\
            np.atan2(state[SE2.Y.value], state[SE2.X.value])

        return predicted_meas

    def gaussian(self, diff, cov):
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2*np.pi)**2 * det_cov)
        return norm_const * np.exp(-0.5 * diff.T @ inv_cov @ diff)
