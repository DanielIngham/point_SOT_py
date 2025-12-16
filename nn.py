from math import nan
import numpy as np

from Simulator.trackSim import Simulator

from sot import SOT

class NN(SOT):
    """docstring for ClassName."""

    def __init__(self, simulator : Simulator):
        super().__init__(simulator)

    def correction(self, state, covariance, measurement_set):

        P_d = self.simulator.detection_prob
        clutter_intense = self.simulator.average_clutter / self.simulator.fov_vol

        max_weight = {
            "w" : 1 - P_d, 
            "v": np.zeros(2), 
            "S": np.zeros((2,2)),
            "H": np.zeros((2,6)),
            "z": (0, 0)
        }

        for measurement in measurement_set:
            v_k = np.array(measurement) - self.predicted_meas(state)
            H_k = self.meas_jacobian(state)
            S_k = H_k @ covariance @ H_k.T + self.R

            weight = P_d * self.gaussian(v_k, S_k) / clutter_intense

            if weight > max_weight["w"]:
                max_weight["w"] = weight
                max_weight["v"] = v_k
                max_weight["S"] = S_k
                max_weight["H"] = H_k
                max_weight["z"] = measurement

        if max_weight["w"] == 1 - P_d: 
            self.choosen_measurements.append((nan,nan))
            return

        self.choosen_measurements.append(max_weight["z"])

        K_k = covariance @ max_weight["H"].T @\
            np.linalg.inv(max_weight["S"]) 

        covariance -= K_k @ max_weight["S"] @ K_k.T
        state += K_k @ max_weight["v"]



if __name__ == "__main__":
    filter = NN(Simulator(1))
    filter.track()
    filter.animation()

