from abc import ABC, abstractmethod
from matplotlib.lines import Line2D
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

from Simulator.trackSim import Simulator
from Simulator.trackSim import SE2
from Simulator.trackSim import Indicies

class SOT(ABC):
    total_states = 6
    total_measurements = 2

    """ 
    Parent Single Object tracking class use for generic tracking functionbality.
    """
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
        """
        Sets the prior (initial) object state equal to the ground truth state created
        by the simulator. 

        Notes
        -----
        It is assumed that the object is detected before tracking commences. 
        """
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

    def prediction_(self, prior : np.ndarray, prior_cov : np.ndarray,
                   posterior : np.ndarray, posterior_cov : np.ndarray) -> None:
        """
        Performs the prediction step on the prior using a constant velocity motion 
        model to get the predicted posterior.  

        Parameters
        ----------
        prior : np.ndarray
            The prior (previous) state of the object.
            Shape: (6,)

        prior_cov : np.ndarray
            The prior (previous) estimation error covariance. This represents the 
            uncertainty in the prior state estimate.

        posterior : np.ndarray
            Vector to store the predicted postior given the prior and the motion model.

        posterior_cov : np.ndarray
            Matrix to store the predicted estimation error covariance.
        """
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
    def correction_(self, state : np.ndarray, covariance : np.ndarray,
                   measurement_set : list) -> None:
        """
        Apply the measurement correction (update) to the predicted state estimate.

        This is an abstract method which must be overwritten by the child class. 
        The child class will be required to perform its own data association to 
        update the state and covariance arguments passed.

        Parameters
        ----------
        state : np.ndarray
            Current predicted state vector.
            Shape: (6,)

        covariance: np.ndarray
            The estimation error covariance of the current state estimate.
            Shape: (6, 6)

        measurement_set: list
            Collection of measurements available for the correction step. 
            Note that this list contains clutter and may not contain an object 
            originated measurement.
        """
        pass

    def track(self) -> None:
        """
        Performs bayesian single object tracking (prediction and correction)
        for all measurements available from the simulation.
        """
        for k in range(1, len(self.simulator.measurements)):
            self.prediction_(self.states[ : , k-1], 
                            self.covariances[k-1], 
                            self.states[ : , k], 
                            self.covariances[k])

            self.correction_(self.states[ : , k], self.covariances[k],
                            self.simulator.measurements[k])

    def motion_jacobian(self, state : np.ndarray) -> np.ndarray: 
        """
        Calculates the Jacobian matrix of motion model given the objects current state. 

        Parameters
        ----------
        state: np.ndarray
            State of the system.
            Shape: (4, )

        Returns
        -------
        Jacobian : np.ndarray
            The Jacobian matrix of the motion model given the current state.
            Shape: (4, 6)
        """
        jacobian = np.zeros((self.total_states, self.total_states))

        theta = state[SE2.O.value]
        v = state[SE2.V.value]

        delta_t = self.sample_rate

        jacobian[0] = [1,0, -v * np.sin(theta) * delta_t, np.cos(theta) * delta_t,0,0]
        jacobian[1] = [0,1,  v * np.cos(theta) * delta_t, np.sin(theta) * delta_t,0,0]
        jacobian[2] = [0,0,1,0,0,0]
        jacobian[3] = [0,0,0,1,0,0]

        return jacobian

    def meas_jacobian(self, state : np.ndarray) -> np.ndarray:
        """
        Calculates the Jacobian matrix of measurement model given the
        objects current state. 

        Parameters
        ----------
        state: np.ndarray
            Predicted state of the system.
            Shape: (4, )

        Returns
        -------
        Jacobian : np.ndarray
            The Jacobian matrix of the measurement model given the 
            current state of the object.
            Shape: (4, 6)
        """
        jacobian = np.zeros((self.total_measurements, self.total_states))
        x = state[SE2.X.value]
        y = state[SE2.Y.value]
        r = np.sqrt(np.pow(x,2) + np.pow(y,2))

        jacobian[0] = [x/r, y/r, 0, 0, 0, 0]
        jacobian[1] = [-y/np.pow(r,2), x/np.pow(r,2), 0, 0, 0, 0]

        return jacobian

    def predicted_meas(self, state) -> np.ndarray:
        """
        Calculates the value of what the measurement would be given the 
        predicted state of the object. 

        Parameters
        ----------
        state: np.ndarray
            Predicted state of the system.
            Shape: (4, )

        Returns
        -------
        prediction : np.ndarray
            The measurement that would be expected given the predicted state of the 
            object.
            Shape: (2,)

        """
        predicted_meas = np.zeros(2, dtype=np.float64)

        predicted_meas[Indicies.RANGE.value] =\
            np.sqrt(np.pow(state[SE2.X.value],2) +\
                np.pow(state[SE2.Y.value],2))

        predicted_meas[Indicies.BEARING.value] =\
            np.atan2(state[SE2.Y.value], state[SE2.X.value])

        return predicted_meas

    def gaussian(self, diff : np.ndarray, cov : np.ndarray) -> np.float64:
        """
        Function for calculating the value of a 2D multivariate Gaussian distribution.

        Parameters
        ----------
        diff : np.ndarray
            The difference between the observed value of the random variable and its 
            expected value (mean).

        cov : np.ndarray
            The covariance matrix of the Gaussian distribution.

        Returns
        -------
        Value of the multivariate Gaussian distribution.

        Note
        ----
        The covariance matrix must be a square matrix.
        The dimensions of the diff vector and cov matrix must be compatible.
        """
        n = diff.shape[0]
        if n != cov.shape[0]:
            raise ValueError("Mismatch between difference vector and covariance matrix\
                sizes")

        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2*np.pi)**n * det_cov)

        return norm_const * np.exp(-0.5 * diff.T @ inv_cov @ diff)

    def get_ellipse(self, mean, cov, n_std=3., **kwargs) -> Ellipse:
        """
        Creates an ellipse class from a 2D multivariate Gaussian distribution used for 
        the animated plotting.

        Parameters
        ----------
        mean : np.ndarray
            The mean of the 2D multivariate Gaussian distribution. It represents the 
            2D x and y position of the center of the ellipse.

        cov : np.ndarray
            The covariance matrix of the multivariate Gaussian distribution.
        
        n_std : float
            The number of standard deviations that the circumference of the ellipse
            should represent.

        **kwargs
            Ellipse class attributes that can be included.

        Returns
        -------
        Instance of an Ellipse class that matches the shape of a 2D multivariate 
        Gaussian distribution.
        """

        # Eigen-decomposition
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Compute ellipse parameters
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = n_std * np.sqrt(vals)  

        # Create ellipse
        return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)


    def plot_meas(self, frame : int, clutter_plot : Line2D, 
                  measurement_plot : Line2D) -> None:
        """
        Plots the measurements recieved at each time step. These measurements 
        include clutter and the groundtruth object originated measurement.

        Parameters
        ----------
        frame : int
            The current frame of the animation.

        clutter_plot : Line2D
            The plot containing all the clutter measurements

        measurement_plot : Line2D
            The plot containing the object originated measurement.
        """
        # Divide the measurments into clutter and object originated lists
        clutter = []
        actual_measurement = [] 
        for measurement in self.simulator.measurements[frame]:
            if measurement in self.simulator.clutter[frame]:
                clutter.append(measurement)
            else:
                actual_measurement = measurement

        # Convert clutter range and bearing points into 2D xy coordintes 
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

    def plot_covariance(self, frame) -> None:
        """
        Adds a new ellipse patch to the plot to indicate the current estimation
        error covariance.

        Parameters
        ----------
        frame : int
            The current frame of the animation.
        """
        cov = self.covariances[frame, :2, :2]
        mean = self.states[ : 2, frame]

        new_ellipse = self.get_ellipse(mean, cov, n_std=3.,
                                         edgecolor='blue', facecolor='none')

        # Copy properties from new_ellipse to existing ellipse
        self.covariance.set_center(new_ellipse.get_center())
        self.covariance.width = new_ellipse.width
        self.covariance.height = new_ellipse.height
        self.covariance.angle = new_ellipse.angle


    def plot_choosen_measurement(self, frame : int) -> None:
        """
        Plots the measurement choosen by the filter.

        Specifically for maximum aposteriori estimates like nearest neighbour, 
        where a single measurement is choosen from the set of measurements containing
        clutter.

        Parameters
        ----------
        frame : int
            The current frame of the animation.

        Notes
        -----
        Other filtering approaches like Probabilistic Data Association (PDA), and
        Gaussian Sum Filtering (GSF) don't explicitly select a measurement, therefore
        they would not have this plot present in their animation.  
        """
        if (len(self.choosen_measurements) > 0):
            self.choosen_meas_plot.set_data([self.x[frame - 1]],[self.y[frame - 1]])

    def update(self, frame : int) -> tuple:
        """
        Updates the animation using data for the current frame.

        Parameters
        ----------
        frame : int
            The current frame of the animation.
        """
        self.plot_meas(frame, self.clutter_plot, self.real_meas_plot)

        self.plot_choosen_measurement(frame)

        self.plot_covariance(frame)

        self.trajectory.set_data(self.states[0, : frame + 1],
                                 self.states[1, :frame + 1]) 


        vibes : list = [[], []]
        for id, trajectory in self.simulator.objects.items():
            vibes[SE2.X.value].append(trajectory[SE2.X.value][frame])
            vibes[SE2.Y.value].append(trajectory[SE2.Y.value][frame])

        self.groundtruth_state.set_data(vibes[SE2.X.value], vibes[SE2.Y.value])

        if (len(self.choosen_measurements) > 0):
            return self.trajectory, self.clutter_plot,\
                self.real_meas_plot, self.choosen_meas_plot, self.covariance
        else:
            return self.trajectory, self.clutter_plot,\
                self.real_meas_plot, self.covariance

    def animation(self) -> None:
        """
        Creates an animation containing:
        - ground truth object track,
        - track estimate produced by the tracking algorithm,
        - track estimation error covariance,  
        - object originated measurement (if present), and
        - clutter measurments.
        """
        
        fig, ax = plt.subplots(figsize=(10, 6))

        self.simulator.plot_trajectories(ax)

        self.groundtruth_state, = ax.plot([], [], 'o' ,
                                          # color='red',
                                 markeredgecolor='red', 
                                 markerfacecolor='none', 
                                 markeredgewidth=2, 
                                 markersize=10,
                                   label="Groundtruth")

        self.trajectory, = ax.plot([], [], "-o" ,
                                   color='blue',
                                   linewidth=2,
                                   label="Trajectory")

        self.clutter_plot, = ax.plot([], [], 'o',
                                     color='grey',
                                     alpha=0.5,
                                     label="Clutter")

        self.real_meas_plot, = ax.plot([], [], 'o',
                                       color='red',
                                       label="True Measurement")

        if (len(self.choosen_measurements) > 0):
            self.choosen_meas_plot, = ax.plot([], [], 'o',
                                              color='green',
                                              alpha=0.5,
                                              label="Selected Measurement")

        cov = self.covariances[0, :2, :2]
        mean = self.states[ : 2, 0]

        self.covariance = self.get_ellipse(mean, cov , n_std=3., 
                              edgecolor='blue', facecolor='none')

        ax.add_patch(self.covariance)

        if (len(self.choosen_measurements) > 0):
            self.xy = [(r * np.cos(b), r * np.sin(b)) for r, b in self.choosen_measurements]
            self.x,self.y = zip(*self.xy)


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
