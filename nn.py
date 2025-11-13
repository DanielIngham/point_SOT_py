import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

from Simulator.trackSim import Simulator
from Simulator.trackSim import SE2
from Simulator.trackSim import Indicies

class NN():
    """docstring for ClassName."""
    total_states = 6
    total_measurements = 2

    def __init__(self, simulator: Simulator):
        # super(ClassName, self).__init__()
        self.simulator : Simulator = simulator
        self.states = np.zeros((self.total_states, len(self.simulator.measurements)), 
                               dtype=np.float64)
        self.covariances = np.zeros((self.simulator.datapoints, 
                                     self.total_states, self.total_states), 
                                    dtype=np.float64)

        self.sample_rate = simulator.sample_rate

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

    def prediction(self, k : int):
        self.states[SE2.X.value, k] =  self.states[SE2.X.value, k - 1] + \
            self.sample_rate * self.states[SE2.V.value, k - 1] * \
                np.cos(self.states[SE2.O.value, k - 1])

        self.states[SE2.Y.value, k] = self.states[SE2.Y.value, k - 1] + \
            self.sample_rate * self.states[SE2.V.value, k - 1] * \
                np.sin(self.states[SE2.O.value, k - 1])

        self.states[SE2.O.value, k] = self.states[SE2.O.value, k - 1]
        self.states[SE2.V.value, k] = self.states[SE2.V.value, k - 1]
        self.states[SE2.A.value, k] = .0
        self.states[SE2.W.value, k] = .0

        F = self.motion_jacobian(self.states[ : , k - 1])
        self.covariances[k] = F @ self.covariances[k - 1] @ F.transpose() + self.Q

    def motion_jacobian(self, state : np.ndarray) -> np.ndarray: 
        jacobian = np.zeros((self.total_states, self.total_states))

        theta = state[SE2.O.value]
        v = state[SE2.V.value]

        delta_t = self.simulator.sample_rate

        jacobian[0] = [1,0, -v * np.sin(theta) * delta_t, np.cos(theta) * delta_t,0,0]
        jacobian[1] = [0,1,  v * np.cos(theta) * delta_t, np.sin(theta) * delta_t,0,0]
        jacobian[2] = [0,0,1,0,0,0]
        jacobian[3] = [0,0,0,1,0,0]

        return jacobian

    def correction(self, k : int):

        P_d = self.simulator.detection_prob
        clutter_intense = self.simulator.average_clutter / self.simulator.fov_vol

        max_weight = {
            "w" : 1 - P_d, 
            "v": np.zeros(2), 
            "S": np.zeros((2,2)),
            "H": np.zeros((2,6))
        }

        for measurement in self.simulator.measurements[k]:
            v_k = np.array(measurement) - self.predicted_meas(k)
            H_k = self.meas_jacobian(self.states[ : , k])
            S_k = H_k @ self.covariances[k] @ H_k.transpose() + self.R

            weight = P_d * self.mvn_pdf(v_k, S_k) / clutter_intense

            if weight > max_weight["w"]:
                max_weight["w"] = weight
                max_weight["v"] = v_k
                max_weight["S"] = S_k
                max_weight["H"] = H_k

        if max_weight["w"] == 1 - P_d: 
            return

        K_k = self.covariances[k] @ max_weight["H"].transpose() @\
            np.linalg.inv(max_weight["S"]) 

        self.covariances[k] -= K_k @ max_weight["S"] @ K_k.transpose()
        self.states[ : , k] += K_k @ max_weight["v"]



    def mvn_pdf(self, diff, cov):
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2*np.pi)**2 * det_cov)
        return norm_const * np.exp(-0.5 * diff.T @ inv_cov @ diff)

    def meas_jacobian(self, state : np.ndarray) -> np.ndarray :
        jacobian = np.zeros((self.total_measurements, self.total_states))
        x = state[SE2.X.value]
        y = state[SE2.Y.value]
        r = np.sqrt(np.pow(x,2) + np.pow(y,2))

        jacobian[0] = [x/r, y/r, 0, 0, 0, 0]
        jacobian[1] = [-y/np.pow(r,2), x/np.pow(r,2), 0, 0, 0, 0]

        return jacobian


    def predicted_meas(self, k) -> np.ndarray:
        predicted_meas = np.zeros(2, dtype=np.float64)

        predicted_meas[Indicies.RANGE.value] =\
            np.sqrt(np.pow(self.states[SE2.X.value][k],2) +\
                np.pow(self.states[SE2.Y.value][k],2))

        predicted_meas[Indicies.BEARING.value] =\
            np.atan2(self.states[SE2.Y.value][k], self.states[SE2.X.value][k])

        return predicted_meas


    def track(self):
        for k in range(1, len(self.simulator.measurements)):
            self.prediction(k)
            self.correction(k)

    def get_ellipse(self, i, n_std=3., **kwargs) :
        cov = self.covariances[i, :2, :2]   # top-left 2x2
        mean = self.states[ : 2, i]

        # Eigen-decomposition
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Compute ellipse parameters
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)  # 2σ ellipse

        # Create ellipse patch
        return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)


    def plot_trajectories(self, ax):
        """
        Plots the trajectory line that the object followed.
        Parameters:
            ax (Axis): matplotlib axis.
        """

        ax.plot(self.states[SE2.X.value, : ],
                self.states[SE2.Y.value, : ], label="NN", marker='o')

        for i in range(len(self.states)):
            if (i % 10 == 0):
                ax.add_patch(self.get_ellipse(i , n_std=3., 
                                      edgecolor='blue', facecolor='none'))


def update(frame):

    clutter = []
    actual_measurement = [] 
    for measurement in filter.simulator.measurements[frame]:
        if measurement in filter.simulator.clutter[frame]:
            clutter.append(measurement)
        else:
            actual_measurement = measurement

    ranges = np.array([r for r, _ in clutter])
    bearings_rad = np.array([b for _, b in clutter]) 

    x_components = ranges * np.cos(bearings_rad)
    y_components = ranges * np.sin(bearings_rad)
    
    scat.set_data(x_components, y_components)
    if actual_measurement != [] :
        scat2.set_data([actual_measurement[0] * np.cos(actual_measurement[1])], 
                       [actual_measurement[0] * np.sin(actual_measurement[1])])
    else:
        scat2.set_data([], [])

    new_ellipse = filter.get_ellipse(frame, n_std=3.,
                                     edgecolor='blue', facecolor='none')

    # Copy properties from new_ellipse → existing ellipse
    ellipse.set_center(new_ellipse.get_center())
    ellipse.width = new_ellipse.width
    ellipse.height = new_ellipse.height
    ellipse.angle = new_ellipse.angle

    line.set_data(filter.states[0, : frame + 1], filter.states[1, :frame + 1]) 
    return line, scat, scat2, ellipse

if __name__ == "__main__":
    filter = NN(Simulator(1))
    filter.track()

    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], color='blue', linewidth=2)
    scat, = ax.plot([], [], 'o', color='blue')
    scat2, = ax.plot([], [], 'o', color='red')

    ellipse = filter.get_ellipse(0 , n_std=3., 
                          edgecolor='blue', facecolor='none')
    ax.add_patch(ellipse)

    # filter.simulator.plot_clutter(ax)
    # filter.plot_trajectories(ax)
    # filter.simulator.plot_trajectories(ax)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(1, len(filter.states[0])),   # one frame per point
        interval=300,        # milliseconds between frames
        repeat=True
    )

    ax.set_xlim(left=filter.simulator.fov[0][0], right=filter.simulator.fov[0][1])
    ax.set_ylim(bottom=filter.simulator.fov[1][0], top=filter.simulator.fov[1][1])
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_title('Plot of Objects from Dictionary')
    # ax.legend(title='Object ID')
    ax.grid(True)

    plt.show()
