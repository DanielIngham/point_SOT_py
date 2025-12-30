import numpy as np
from typing import TypedDict

from Simulator.trackSim import Simulator
from sot import SOT


class Hypothesis(TypedDict):
    max : int                   # Maximum number of hypothesis maintained. 
    total : int                 # Current number of hypothesis being maintained.
    state : list[np.ndarray]    # List of states for each hypothesis.
    cov : list[np.ndarray]      # List of covariances for each hypothesis.
    weight : list[float]   # List of weights of each hypothesis. 

class GSF(SOT):
    """
    Gaussian Sum Filter.
    """

    def __init__(self, max_hypothesis : int, simulator : Simulator):
        super().__init__(simulator)

        self.hypothesis : Hypothesis = {
            "max" : max_hypothesis,
            "total" : 0,
            "state" : [],
            "cov" : [],
            "weight" : [],
        }

        self.hypothesis["total"] += 1
        self.hypothesis["state"].append(self.states[ : , 0])
        self.hypothesis["cov"].append(self.covariances[0])
        self.hypothesis["weight"].append(1.)

    def track(self) -> None:
        """
        Performs bayesian single object tracking (prediction and correction)
        for all measurements available from the simulation.
        """
        for k in range(1, len(self.simulator.measurements)):
            # Propagate the hypotheses.
            self.states[ : , k], self.covariances[k] = self.prediction()
            self.correction_(self.states[ : , k], self.covariances[k],
                            self.simulator.measurements[k])

    def prediction(self) -> tuple[np.ndarray, np.ndarray] :

        for i in range(self.hypothesis["total"]) :
            self.hypothesis["state"][i], self.hypothesis["cov"][i] = self.prediction_(
                self.hypothesis["state"][i], self.hypothesis["cov"][i])

        # Return the hypothesis with the maximum weight (MAP estimate)
        weights = self.hypothesis["weight"]
        index = weights.index(max(weights))

        return self.hypothesis["state"][index], self.hypothesis["cov"][index]

    def correction(self):
        for i in range(self.hypothesis["total"]) :
            pass

    def correction_(self, state, covariance, measurement_set):
        pass

if __name__ == "__main__":
    filter = GSF(max_hypothesis=5, simulator=Simulator(1, 42))
    filter.track()
    filter.animation()
