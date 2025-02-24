import numpy as np
import sklearn.gaussian_process as gpr
import warnings

warnings.simplefilter("ignore", DeprecationWarning)


def gp_clogs(train, x_data=None, y_data=None, model=None, x=None, mu=None, sigma=None):
    """Train or predict using a Gaussian Process model for clogging data.
    This function either trains a Gaussian Process model using provided training data,
    or generates predictions using a pre-trained model.
    """
    # Train the model
    if train:
        mu = np.mean(x_data)
        sigma = np.std(x_data)
        x_norm = (x_data - mu) / sigma
        kernel = gpr.kernels.Matern(length_scale=5, length_scale_bounds=(1, 10), nu=1.5)
        model = gpr.GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=10, normalize_y=True
        )
        model.fit(x_norm.reshape(-1, 1), y_data.reshape(-1, 1))
        return model, mu, sigma

    # generate predictions
    else:
        x_norm = (x - mu) / sigma
        return model.predict(x_norm.reshape(-1, 1))


def heat_sink(u, j, q_x, q_y, Nx, T_0):
    """
    Calculate the heat sink effect of the cooling water on the belt in the pastillation process.

    This function models the heat transfer between the cooling water and the belt at specific
    points in the discretized grid, accounting for the temperature difference and material
    properties.

    Parameters
    ----------
    u : ndarray
        Temperature field matrix representing the current temperature distribution.
    j : int
        Current row index in the temperature field matrix.
    q_x : ndarray
        Array of x-coordinates where cooling nozzles are located.
    q_y : ndarray
        Array of y-coordinates where cooling nozzles are located.
    Nx : int
        Number of points in x-direction (width of the temperature field matrix).
    T_0 : float
        Reference temperature offset (in Fahrenheit).
    """
    f_vec = np.zeros(Nx)
    rho_belt = 7850
    # density of belt (carbon steel, kg/m3)

    V_belt = 0.0234**2 * 0.01 * 2
    # volume of belt over nozzle assuming pastille diameter of 2.34 cm and belt thickness of 10 mm

    Cp_belt = 466  # heat capacity of belt (carbon steel, J/kg/K)
    m_water = 4 * 3.63 / 60 / 2 / 189  # flowrate of water (kg/timestep)
    T_water = 24.5  # initial temperature of cooling water (deg C)
    if j in q_y:
        f_vec[q_x] = (
            -m_water
            * 4148
            * (5 / 9 * (u[j, q_x] + T_0 - 32) - T_water)
            / (rho_belt * V_belt * Cp_belt)
        )
    return f_vec
