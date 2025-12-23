from Simulator.trackSim import Simulator
from sot import SOT

class GSF(SOT):
    """
    Gaussian Sum Filter.
    """
    def __init__(self, simulator : Simulator):
        super().__init__(simulator)

        self.hypothesis = {
            "state" : [],
            "covariance" : [],
            "weight" : [],
        }

    def correction(self, state, covariance, measurement_set):
        pass

if __name__ == "__main__":
    filter = GSF(Simulator(1, 42))
    filter.track()
    filter.animation()
