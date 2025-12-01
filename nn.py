from math import nan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

from Simulator.trackSim import Simulator
from Simulator.trackSim import SE2
from Simulator.trackSim import Indicies

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

        covariance[:] -= K_k @ max_weight["S"] @ K_k.T
        state[:] += K_k @ max_weight["v"]

    def get_ellipse(self, mean, cov, n_std=3., **kwargs) :

        # Eigen-decomposition
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Compute ellipse parameters
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)  # 2σ ellipse

        # Create ellipse patch
        return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)


    def plot_meas(self, frame, clutter_plot, measurement_plot): 
        clutter = []
        actual_measurement = [] 
        for measurement in self.simulator.measurements[frame]:
            if measurement in self.simulator.clutter[frame]:
                clutter.append(measurement)
            else:
                actual_measurement = measurement

        ranges = np.array([r for r, _ in clutter])
        bearings_rad = np.array([b for _, b in clutter]) 

        x_components = ranges * np.cos(bearings_rad)
        y_components = ranges * np.sin(bearings_rad)

        clutter_plot.set_data(x_components, y_components)

        if actual_measurement != [] :
            measurement_plot.set_data(
                [actual_measurement[0] * np.cos(actual_measurement[1])],
                [actual_measurement[0] * np.sin(actual_measurement[1])])
        else:
            measurement_plot.set_data([], [])

    def plot_covariance(self, frame):
        cov = self.covariances[frame, :2, :2]
        mean = self.states[ : 2, frame]

        new_ellipse = self.get_ellipse(mean, cov, n_std=3.,
                                         edgecolor='blue', facecolor='none')

        # Copy properties from new_ellipse to existing ellipse
        self.covariance.set_center(new_ellipse.get_center())
        self.covariance.width = new_ellipse.width
        self.covariance.height = new_ellipse.height
        self.covariance.angle = new_ellipse.angle


    def plot_choosen_measurement(self, frame):
        if (len(self.choosen_measurements) > 0):
            self.choosen_meas_plot.set_data([self.x[frame - 1]],[self.y[frame - 1]])

    def update(self, frame):
        self.plot_meas(frame, self.clutter_plot, self.real_meas_plot)
        self.plot_choosen_measurement(frame)

        self.plot_covariance(frame)

        self.trajectory.set_data(self.states[0, : frame + 1],
                                 self.states[1, :frame + 1]) 

        return self.trajectory, self.clutter_plot, \
            self.real_meas_plot, self.choosen_meas_plot, self.covariance
    
    def animation(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.trajectory, = ax.plot([], [], "-o" ,color='blue', linewidth=2,
                                   label="Trajectory")
        self.clutter_plot, = ax.plot([], [], 'o', color='grey', alpha=0.5,
                                     label="Clutter")
        self.real_meas_plot, = ax.plot([], [], 'o', color='red',
                                       label="True Measurement")
        self.choosen_meas_plot, = ax.plot([], [], 'o', color='green', alpha=0.5,
                                          label="Selected Measurement")

        cov = self.covariances[0, :2, :2]
        mean = self.states[ : 2, 0]

        self.covariance = self.get_ellipse(mean, cov , n_std=3., 
                              edgecolor='blue', facecolor='none')

        ax.add_patch(self.covariance)

        if (len(self.choosen_measurements) > 0):
            self.xy = [(r * np.cos(b), r * np.sin(b)) for r, b in self.choosen_measurements]
            self.x,self.y = zip(*self.xy)

        self.simulator.plot_trajectories(ax)

        ani = animation.FuncAnimation(
            fig,
            self.update,
            frames=range(1, len(self.states[0])),
            interval=500,
            repeat=True
        )

        ax.set_xlim(left=self.simulator.fov[0][0], right=self.simulator.fov[0][1])
        ax.set_ylim(bottom=self.simulator.fov[1][0], top=self.simulator.fov[1][1])
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Single Object Tracking')
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.show()


if __name__ == "__main__":
    filter = NN(Simulator(1))
    filter.track()
    filter.animation()

