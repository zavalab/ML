import os
import pickle
import itertools
from tqdm import tqdm
import numpy as np
from scipy import sparse
from pastillation.data.helping import gp_clogs, heat_sink

DATA_DIR = "/Pastillation/data/"


def gen_data(control_parameters, time_parameters, pellet_temps, tune=False):
    """
    Generate temperature and flow data for a pastillation process simulation.

    This function simulates the temperature distribution and material flow in a 2D grid representing
    a pastillation system, incorporating PID control for temperature regulation.

    Parameters
    ----------
    control_parameters : array-like
        List/array containing PID control parameters:
        [K_p (proportional gain), t_I (integral time), t_d (derivative time),
         T_sp (temperature setpoint), u_bar (initial belt speed)]
    time_parameters : array-like 
        List/array containing time discretization parameters:
        [T (total simulation time), dt (time step), theta (implicitness parameter)]
    pellet_temps : array-like
        Temperature values of pellets used for GP model training
    tune : bool, optional
        If True, returns error history starting from when first row reaches end of domain.
        Default is False.

    Returns
    -------
    u : ndarray
        3D array of temperature values with shape (Ny, Nx, Nt)
    flow : ndarray
        3D array of flow values with shape (Ny, Nx, Nt)
    T_obs : ndarray
        1D array of observed temperatures over time
    F_obs : ndarray
        1D array of observed flow rates over time
    """
    args = gen_args()
    space_parameters = args[0]
    # time_parameters = args[2]
    alpha = args[3]
    T_0 = args[4]
    # pellet_temps = args[5]

    Lx = space_parameters[0]
    Nx = space_parameters[1]
    nozzle_x_spacing = space_parameters[2]
    Ly = space_parameters[3]
    Ny = space_parameters[4]
    nozzle_y_spacing = space_parameters[5]
    N = (Nx) * (Ny)
    m = lambda i, j: j * (Nx) + i

    T = time_parameters[0]
    dt = time_parameters[1]
    theta = time_parameters[2]

    K_p = control_parameters[0]
    t_I = control_parameters[1]
    t_d = control_parameters[2]
    T_sp = control_parameters[3]
    u_bar = int(control_parameters[4])

    x = np.linspace(0, Lx - 1, Nx)
    y = np.linspace(0, Ly - 1, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Nt = int(T / dt) + 1
    t = np.linspace(0, T, Nt)

    Ix = range(0, Nx)
    Iy = range(0, Ny)
    It = range(0, Nt)

    Fx = alpha * dt / (dx**2)
    Fy = alpha * dt / (dy**2)

    error_history_temp = np.zeros(T)
    speed_profile = u_bar * np.ones(T, dtype=int)
    F_obs = np.ones(T)
    T_obs = np.ones(T)
    ROW = np.ones(T)

    # Mass flow tracking setup and initial conditions (t = 0)
    flow_in = np.zeros(Nx)
    flow_n = np.zeros((Ny, Nx))
    flow = np.zeros((Ny, Nx, Nt))

    flow[:, :, 0] = flow_n

    # Temperature tracking setup initial conditions (t = 0)
    u_n = np.zeros((Ny, Nx))
    u = np.zeros((Ny, Nx, Nt))

    u[:, :, 0] = u_n

    DsDt = np.zeros(Nt - 1)
    outflow = []

    # setup of system of sparse linear system
    b = np.zeros(N)

    inferior = np.zeros(N - (Nx))  # sub-lower diagonal
    lower = np.zeros(N - 1)  # lower diagonal
    main = np.zeros(N)  # main diagonal
    upper = np.zeros(N - 1)  # upper diagonal
    superior = np.zeros(N - (Nx))  # super-upper diagonal

    ## top boundary
    j = 0
    main[m(0, j) : m(Nx, j)] = 1

    for j in Iy[1:-1]:
        # left boundary
        i = 0
        main[m(i, j)] = 1
        # right boundary
        i = Nx - 1
        main[m(i, j)] = 1
        # interior points
        superior[m(1, j) : m(Nx - 1, j)] = -theta * Fy
        upper[m(1, j) : m(Nx - 1, j)] = -theta * Fx

        main[m(1, j) : m(Nx - 1, j)] = 1 + 2 * theta * (Fx + Fy)

        lower[m(1, j) - 1 : m(Nx - 1, j) - 1] = -theta * Fx
        inferior[m(1, j) - (Nx) : m(Nx - 1, j) - (Nx)] = -theta * Fy

    ## bottom boundary
    j = Ny - 1
    main[m(0, j) : m(Nx, j)] = 1

    ## build sparse coefficient matrix A
    A = sparse.diags(
        diagonals=[main, lower, upper, inferior, superior],
        offsets=[0, -1, 1, -(Nx), Nx],
        shape=(N, N),
        format="csr",
    )

    # Solve system of equations Ac = b
    for n in tqdm(It[0:-1], total=len(It[0:-1])):
        ## set sinks (water nozzles) as forcing function that cover a 3x3 square centered on the nozzle
        if n == 0:
            ### x-coordinates of spaces in contact with water
            q_x = np.arange(0, Nx, nozzle_x_spacing, dtype=int)
            q_x = q_x[1:-1]
            # q_x = np.hstack([q_x-1, q_x, q_x+1])

            ### y-cordinates of spaces in contact with water
            q_y = np.arange(0, Ny, nozzle_y_spacing, dtype=int)
            q_y = q_y[1:]
            q_y = np.hstack([q_y - 1, q_y, q_y + 1])

        ## train model for predicting clogs
        if n == 0:
            loc = np.linspace(0, len(pellet_temps) - 1, len(pellet_temps))
            model, mu, sigma = gp_clogs(True, x_data=loc, y_data=pellet_temps)
            position = []

        ## fill in temperature values along top boundary
        j = 0
        i = 0
        p = m(i, j)
        b[p] = 0
        k = -1

        position.append(0)

        for i in Ix[1:-1]:
            p = m(i, j)

            if (i + 1) % 2 == 0:
                k += 1
                source = gp_clogs(
                    False, model=model, x=np.array([k]), mu=mu, sigma=sigma
                )
                seed = np.random.rand()
                source = max(0, np.sign(source - seed))
                flow_in[i] = source
                source = 140 * source
                b[p] = source

            else:
                flow_in[i] = 0
                b[p] = 0

            flow_n[j, i] = flow_in[i]
            u_n[j, i] = b[p]

        i = Nx
        p = m(i, j)
        b[p] = 0

        ## calculate function values for internal y coordinate points
        for j in Iy[1:-1]:
            ### middle rows, left boundary
            i = 0
            p = m(i, j)
            b[p] = 0

            ### interior mesh points
            i_min = Ix[1]
            i_max = Ix[-1]

            if n == 0:
                b[m(i_min, j) : m(i_max, j)] = (
                    u_n[j, i_min:i_max]
                    + (1 - theta)
                    * (
                        Fx
                        * (
                            u_n[j, i_min - 1 : i_max - 1]
                            - 2 * u_n[j, i_min:i_max]
                            + u_n[j, i_min + 1 : i_max + 1]
                        )
                        + Fy
                        * (
                            u_n[j - 1, i_min:i_max]
                            - 2 * u_n[j, i_min:i_max]
                            + u_n[j + 1, i_min:i_max]
                        )
                    )
                    + theta * dt * heat_sink(u_n, j, q_x, q_y, Nx, T_0)[i_min:i_max]
                    + (1 - theta)
                    * dt
                    * heat_sink(u_n, j, q_x, q_y, Nx, T_0)[i_min:i_max]
                )
            else:
                b[m(i_min, j) : m(i_max, j)] = (
                    u_n[j, i_min:i_max]
                    + (1 - theta)
                    * (
                        Fx
                        * (
                            u_n[j, i_min - 1 : i_max - 1]
                            - 2 * u_n[j, i_min:i_max]
                            + u_n[j, i_min + 1 : i_max + 1]
                        )
                        + Fy
                        * (
                            u_n[j - 1, i_min:i_max]
                            - 2 * u_n[j, i_min:i_max]
                            + u_n[j + 1, i_min:i_max]
                        )
                    )
                    + theta * dt * heat_sink(u_n, j, q_x, q_y, Nx, T_0)[i_min:i_max]
                    + (1 - theta)
                    * dt
                    * heat_sink(u_n, j, q_x, q_y, Nx, T_0)[i_min:i_max]
                )

            ### middle rows right boundary
            i = Nx - 1
            p = m(i, j)
            b[p] = 0

        ### bottom boundary
        j = Ny - 1
        b[m(0, j) : m(Nx, j)] = u_n[Ny - 1, :]

        ## solve system of linear equations
        c = sparse.linalg.spsolve(A, b)

        ## calculate error between average pellet row temperature and desired setpoint and select new control action

        if n == 0:
            F_obs[n] = np.sum(flow_n) / Ny * u_bar
            T_obs[n] = np.mean(c.reshape(Ny, Nx)[n])
            error_temp = T_sp - T_obs[n]
            error_history_temp[n] = error_temp

            dsdt = K_p * (
                error_temp
                - error_history_temp[n - 1]
                + dt / t_I * error_temp
                + t_d
                * (
                    error_temp
                    - 2 * error_history_temp[n - 1]
                    + error_history_temp[n - 2]
                )
                / dt
            )
            dsdt = min(max(-1, dsdt), 1)
            speed = int(np.round(speed_profile[n - 1] + dsdt, 0))
            speed = min(max(2, speed), 12)

            DsDt[n] = dsdt
            speed_profile[n] = speed
            row = position[0]

        else:
            row = position[0]

            if row > Ny - 1:
                position_dummy = []
                idx = np.where(np.array(position) > Ny - 1)[0]
                past_end = np.array(position)[idx] - speed
                position_dummy = [
                    position[i] for i, _ in enumerate(position) if i not in idx
                ]
                position = position_dummy.copy()
                row = position[0]
                outflow.append(np.sum(flow[:, :, n][past_end]))

                del position_dummy

            F_obs[n] = np.sum(flow_n) / Ny * speed
            T_obs[n] = np.mean(c.reshape(Ny, Nx)[row])
            error_temp = T_sp - T_obs[n]
            error_history_temp[n] = error_temp

            dsdt = K_p * (
                error_temp
                - error_history_temp[n - 1]
                + dt / t_I * error_temp
                + t_d
                * (
                    error_temp
                    - 2 * error_history_temp[n - 1]
                    + error_history_temp[n - 2]
                )
                / dt
            )
            dsdt = min(max(-1, dsdt), 1)
            speed = int(np.round(speed_profile[n - 1] + dsdt, 0))
            speed = min(max(2, speed), 12)

            DsDt[n] = dsdt
            speed_profile[n] = speed

        ## fill in flow and update flow_n
        flow[:, :, n + 1] = flow_n.copy()
        flow_n[speed:, :] = flow_n[:-speed, :]
        flow_n[:speed] = 0

        ## fill in u and update u_n
        u[:, :, n + 1] = c.reshape(Ny, Nx)
        u_n = u[:, :, n + 1].copy()
        u_n[speed:, :] = u_n[:-speed, :]
        u_n[:speed] = 0

        # update position tracker
        position[:] = [p + speed for p in position]
        ROW[n] = position[0]

    if tune:
        first_row = np.where(ROW >= Ny - 1)[0][0]
        error_history_temp = error_history_temp[first_row:]

    return u, flow, T_obs, F_obs


def gen_fixed_speed(seed):
    """
    Generates synthetic data for fixed speed conditions with PID control parameters.

    This function creates temperature and flow data based on PID control parameters
    and random pellet temperatures. The data is saved to a pickle file.
    """
    K_p = 0
    t_I = np.inf
    t_d = 0

    T_sp = 14
    u_bar = 1

    control_parameters = [K_p, t_I, t_d, T_sp, u_bar]

    dt = 1

    T = 1400

    theta = 1

    time_parameters = [T, dt, theta]

    pellet_temps = np.random.RandomState(seed).random(32)

    temp, flow, temp_obs, flow_obs = gen_data(
        control_parameters, time_parameters, pellet_temps
    )

    flow_obs = flow_obs * u_bar

    temp = temp.transpose((2, 0, 1))
    flow = flow.transpose((2, 0, 1))

    idx = np.random.RandomState(seed).choice(np.arange(0, T, 1), 100, replace=False)

    temp = temp[idx + 1][..., None]
    flow = flow[idx + 1][..., None]
    temp_obs = temp_obs[idx][..., None]
    flow_obs = flow_obs[idx][..., None]

    output_dir = os.path.join(DATA_DIR, f"fixed_speed_{seed}.pickle")

    with open(output_dir, "wb") as handle:
        pickle.dump(temp, handle)
        pickle.dump(temp_obs, handle)
        pickle.dump(flow, handle)
        pickle.dump(flow_obs, handle)

    print("# Saved: ", output_dir)


def gen_varied_speed(seed):
    """
    Generates synthetic data for varied speed conditions with PID control parameters.

    This function creates temperature and flow data based on PID control parameters
    and random pellet temperatures. The data is saved to a pickle file.
    """
    PID_BANK = np.array(
        [
            [
                22.3199118,
                14.436749673951871,
                8.040614190629844,
                0.5000000536613557,
                0.111599559,
            ],
            [
                9.45040648,
                9.363566084085795,
                7.91124063399182,
                7.749975697380466,
                37.80162592,
            ],
            [
                0.0147050843,
                0.007353707548632869,
                0.041991089848582436,
                0.7895536652845879,
                0.002450847383333333,
            ],
        ]
    )
    K_p = PID_BANK[0, 0]
    t_I = PID_BANK[1, 0]
    t_d = PID_BANK[2, 0]

    T_sp = 14 + np.random.RandomState(seed).randn() * 5
    u_bar = 1

    control_parameters = [K_p, t_I, t_d, T_sp, u_bar]

    dt = 1

    T = 1400
    theta = 1

    time_parameters = [T, dt, theta]

    pellet_temps = np.random.RandomState(seed).random(32)

    temp, flow, temp_obs, flow_obs = gen_data(
        control_parameters, time_parameters, pellet_temps
    )

    flow_obs = flow_obs * u_bar

    idx = np.concatenate(
        [
            np.arange(0, 201, 1),
            np.random.RandomState(seed).choice(
                np.arange(202, T, 1), 100, replace=False
            ),
        ]
    )

    temp = temp.transpose((2, 0, 1))
    flow = flow.transpose((2, 0, 1))

    temp = temp[idx + 1][..., None]
    flow = flow[idx + 1][..., None]
    temp_obs = temp_obs[idx][..., None]
    flow_obs = flow_obs[idx][..., None]

    output_dir = os.path.join(DATA_DIR, f"varied_speed_{seed}.pickle")

    with open(output_dir, "wb") as handle:
        pickle.dump(temp, handle)
        pickle.dump(temp_obs, handle)
        pickle.dump(flow, handle)
        pickle.dump(flow_obs, handle)

    print("# Saved: ", output_dir)


def gen_args():
    """
    Lx is the length in the x direction of the belt; it is assumed that each pellet row has 32 pellets with a gap of one pellet
    length around each for a total length of 2*(32)+1 = 65
    Ly is the length in the y direction; it is assumed to be the length of belt required to accomodate 48 rows of water nozzles,
    each with a spacing of 13 pellet lengths for a total length of (48+1)*13 = 637
    Nx and Ny are set to the number of pellet lengths in Lx and Ly
    dt is assumed to be equivalent to 0.5 seconds as the roller is assumed to place down pastilles at a rate of 1 row per second;
    since there is a gap between the rows, each row is placed every other timestep, meaning two pass before a new is placed
    T is the number of timesteps in the simulation
    alpha is manually tuned to give a reasonable temperature profile
    """
    Lx = 65
    Nx = 65
    nozzle_x_spacing = 1
    Ly = 637
    Ny = 637
    nozzle_y_spacing = 13
    space_parameters = [Lx, Nx, nozzle_x_spacing, Ly, Ny, nozzle_y_spacing]

    dt = 1
    T = 637
    theta = 1
    time_parameters = [T, dt, theta]

    PID_BANK = np.array(
        [
            [
                22.3199118,
                14.436749673951871,
                8.040614190629844,
                0.5000000536613557,
                0.111599559,
            ],
            [
                9.45040648,
                9.363566084085795,
                7.91124063399182,
                7.749975697380466,
                37.80162592,
            ],
            [
                0.0147050843,
                0.007353707548632869,
                0.041991089848582436,
                0.7895536652845879,
                0.002450847383333333,
            ],
        ]
    )
    K_p = PID_BANK[0, 0]
    t_I = PID_BANK[1, 0]
    t_d = PID_BANK[2, 0]
    T_sp = 14
    u_bar = 2

    control_parameters = [K_p, t_I, t_d, T_sp, u_bar]

    alpha = 5.95 * 10 ** (-3)
    T_0 = 72  # base or 0 temperature based on thermal camera
    pellet_temps = np.loadtxt(os.path.join(DATA_DIR, "pellet_temperatures_scaled.txt"))

    args = (
        space_parameters,
        [T_sp, u_bar],
        time_parameters,
        alpha,
        T_0,
        pellet_temps,
        True,
        True,
    )

    return args


def job_array(idx, max_idx):
    seeds = np.arange(50).astype("int")
    funcs = [gen_fixed_speed, gen_varied_speed]

    combs = itertools.product(seeds, funcs)
    combs = list(combs)

    size = len(combs) // max_idx
    start = idx * size
    end = min((idx + 1) * size, len(combs))

    return combs[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1
    combs = job_array(idx, max_idx)

    for comb in combs:
        seed, func = comb
        func(seed)
