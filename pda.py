import numpy as np
from numpy._core.numeric import dtype
from numpy.linalg import outer

from sot import SOT
from Simulator.trackSim import Simulator

class PDA(SOT):
    """
    Probabilistic Data Association Filter.
    """
    def __init__(self, simulator : Simulator):
        super().__init__(simulator)

    def correction_(self, state, covariance, measurement_set):
        total_hypothesis = len(measurement_set) + 1

        P_d = self.simulator.detection_prob
        clutter_intense = self.simulator.average_clutter / self.simulator.fov_vol

        predicted_state = state
        predicted_cov = covariance

        """ 
        Create an empty array to store the updated state and covariance 
        for each measurement. 
        """
        possible_states = np.zeros((total_hypothesis, self.total_states), 
                                dtype=np.float64)

        weights = np.zeros(total_hypothesis, dtype=np.float64)

        """
        Set the null hypothesis to the hypothesis that the object was not detected.
        """
        possible_states[0] = predicted_state
        weights[0] = 1 - P_d

        """
        Calculate the updated state and error covariance for each measurement.
        """
        # Measurement Jacobian
        H_k = self.meas_jacobian(predicted_state)
        # Innovation Covariance
        S_k = H_k @ covariance @ H_k.T + self.R
        # Kalman Gain
        K_k = covariance @ H_k.T @ np.linalg.inv(S_k) 

        updated_cov = predicted_cov - K_k @ S_k @ K_k.T


        for i, measurement in enumerate(measurement_set):
            # Innovation
            v_k = np.array(measurement) - self.predicted_meas(predicted_state)

            possible_states[i + 1] = predicted_state + K_k @ v_k
            weights[i + 1] = P_d * self.gaussian(v_k, S_k) / clutter_intense

        """
        Normalise the weights.
        """
        weights = weights / sum(weights)

        """
        Calculate the merged Gaussian density of all weighted hypotehsis.
        """
        state[:] = weights[0] * possible_states[0]

        for i in range(1, total_hypothesis):
            state[:] += weights[i] * possible_states[i]

        diff = state - predicted_state
        covariance[:] = weights[0] * (predicted_cov +  np.outer(diff, diff))

        for i in range(1, total_hypothesis):
            diff = state - possible_states[i]
            covariance[:] += weights[i] * (updated_cov + np.outer(diff, diff))


if __name__ == "__main__":
    filter = PDA(Simulator(total_objects=1, seed=42))
    filter.track()
    filter.animation()

