import numpy as np
import itertools

import gudhi as gd


# Threshold values for a simple example (integers are 1-8)
threshold_values = range(10)

# Creating holders for array values
q1_st = np.zeros(len(threshold_values))
q1_end = np.zeros(len(threshold_values))
q2_st = np.zeros(len(threshold_values))
q2_end = np.zeros(len(threshold_values))
qd_st = np.zeros(len(threshold_values))
qd_end = np.zeros(len(threshold_values))
q3_st = np.zeros(len(threshold_values))
q3_end = np.zeros(len(threshold_values))
q4_st = np.zeros(len(threshold_values))
q4_end = np.zeros(len(threshold_values))

# bitquad map in matrix form
# bitquad_type_mat = np.zeros((len(threshold_values), 5*2))
bitquad_row_map = {'q1_st': 0, 'q1_end': 1, 'q2_st': 2, 'q2_end': 3,
                   'qd_st': 4, 'qd_end': 5, 'q3_st': 6, 'q3_end': 7,
                   'q4_st': 8, 'q4_end': 9}

iter_vals = itertools.product(range(1), repeat=2)
values = []
for i in iter_vals:
    if len(values) == 0:
        values = [np.array(i), ]
    else:
        values.append(i)


def gudhi_get_EC_2D(data, num=1001, filt_start=0, filt_stop=1):
    """
    Function to get EC curve for given data.

    :param data: np.ndarray, N-D array of data
    :param num: int, number of evenly spaced samples
    :param filt_start: float, lower bound of samples
    :param filt_stop: float, upper bounds of samples
    :return: ec: Euler Characteristic curve as a function of sample points
    """
    filtrations = np.linspace(filt_start, filt_stop, num)
    cubeplex = gd.CubicalComplex(dimensions=[data.shape[1], data.shape[0]],
                                 top_dimensional_cells=np.ndarray.flatten(data))
    cubeplex.persistence()
    b = np.zeros((num, 2))
    ec = np.zeros(num)
    # for i, fval in enumerate(np.flip(filtrations)):
    for i, fval in enumerate(filtrations):
        betti = cubeplex.persistent_betti_numbers(fval, fval)
        b[i] = [betti[0], betti[1]]
        ec[i] = betti[0] - betti[1]
    return ec


def gudhi_get_EC_3D(data, num=1001, filt_start=0, filt_stop=1):
    """
    Function to get EC curve for given data.

    :param data: np.ndarray, N-D array of data
    :param num: int, number of evenly spaced samples
    :param filt_start: float, lower bound of samples
    :param filt_stop: float, upper bounds of samples
    :return: ec: Euler Characteristic curve as a function of sample points
    """
    filtrations = np.linspace(filt_start, filt_stop, num)
    cubeplex = gd.CubicalComplex(dimensions=[data.shape[2], data.shape[1], data.shape[0]],
                                 top_dimensional_cells=np.ndarray.flatten(data))
    cubeplex.persistence()
    b = np.zeros((num, 3))
    ec = np.zeros(num)
    # for i, fval in enumerate(np.flip(filtrations)):
    for i, fval in enumerate(filtrations):
        betti = cubeplex.persistent_betti_numbers(fval, fval)
        b[i] = [betti[0], betti[1], betti[2]]
        ec[i] = betti[0] - betti[1] + betti[2]
    return ec

def compute_local_EC_2D(data, bq_map, locs, max_val):
    vals_ind = np.argsort(data)
    # vals_ind = sorted(range(len(data)), key=data.__getitem__)
    blank_mat = bq_map

    # Compute adjacency of 2-pixel on configuration
    # adj = (np.linalg.norm(np.array(locs[vals_ind[0]]) - np.array(locs[vals_ind[1]])) == 1)

    adj = round(np.abs((locs[vals_ind[0]][0] - locs[vals_ind[1]][0])) +
                np.abs((locs[vals_ind[0]][1] - locs[vals_ind[1]][1]))) == 1

    # Fully filled in pixel space (4-pixels) starts at 0 and
    # Ends when we pass A (i.e., A + 1).
    blank_mat[0, 8] += 1
    blank_mat[int(data[vals_ind[0]] + 1), 9] += 1

    # When only 3-pixels are filled. This starts after A (i.e., A + 1)
    # and ends at B (i.e., B + 1)
    blank_mat[int(data[vals_ind[0]] + 1), 6] += 1
    blank_mat[int(data[vals_ind[1]] + 1), 7] += 1
    # q_2 if adjacent, q_d if not
    # Starts after B (i.e., B + 1), and ends after C (i.e., C + 1)
    if adj:
        blank_mat[int(data[vals_ind[1]] + 1), 2] += 1
        blank_mat[int(data[vals_ind[2]] + 1), 3] += 1
    else:
        blank_mat[int(data[vals_ind[1]] + 1), 4] += 1
        blank_mat[int(data[vals_ind[2]] + 1), 5] += 1

    # Single pixel starts after C (i.e., C + 1), and ends after D (i.e., D + 1)
    blank_mat[int(data[vals_ind[2]] + 1), 0] += 1
    blank_mat[int(data[vals_ind[3]] + 1), 1] += 1

    return blank_mat


def compute_local_EC_3D(data, bq_map, locs, max_val):
    vals_ind = np.argsort(data)

    blank_mat = bq_map

    sqrt1 = 1 ** 0.5
    sqrt2 = 2 ** 0.5
    sqrt3 = 3 ** 0.5

    ######################################################################
    # Find the adjacency of two active voxels for Q2i
    adj2 = np.linalg.norm(locs[vals_ind[6]] - locs[vals_ind[7]])

    # Add third voxel to "total adjacency"
    adj3 = adj2 + \
           np.linalg.norm(locs[vals_ind[5]] - locs[vals_ind[6]]) + \
           np.linalg.norm(locs[vals_ind[5]] - locs[vals_ind[7]])

    # Add fourth voxel to "total adjacency"
    adj4 = adj3 + \
           np.linalg.norm(locs[vals_ind[4]] - locs[vals_ind[5]]) + \
           np.linalg.norm(locs[vals_ind[4]] - locs[vals_ind[6]]) + \
           np.linalg.norm(locs[vals_ind[4]] - locs[vals_ind[7]])

    # Uneconomical to compute fifth adjacency, but we can determine the
    # adjacency of the unactive voxels which will still allow us to
    # know the adjacency of the 5-voxel set.
    # Compute the 2 empty voxel adjacency
    adj2n = np.linalg.norm(locs[vals_ind[1]] - locs[vals_ind[0]])

    adj3n = adj2n + \
            np.linalg.norm(locs[vals_ind[2]] - locs[vals_ind[1]]) + \
            np.linalg.norm(locs[vals_ind[2]] - locs[vals_ind[0]])
    ######################################################################

    # Adding Q_81 for when everything is filled in
    blank_mat[0, 40] += 1
    blank_mat[int(data[vals_ind[0]] + 1), 41] += 1

    # Adding Q_71 for when everything is filled in minus one voxel
    blank_mat[int(data[vals_ind[0]] + 1), 38] += 1
    blank_mat[int(data[vals_ind[1]] + 1), 39] += 1

    # Adding Q_6i pieces. Depending on adjacency of 2 empty voxels (adj2n), we add to 61, 62, or 63
    if adj2n >= 1.5:
        blank_mat[int(data[vals_ind[1]] + 1), 36] += 1
        blank_mat[int(data[vals_ind[2]] + 1), 37] += 1
    elif adj2n >= 1.2:
        blank_mat[int(data[vals_ind[1]] + 1), 34] += 1
        blank_mat[int(data[vals_ind[2]] + 1), 35] += 1
    else:
        blank_mat[int(data[vals_ind[1]] + 1), 32] += 1
        blank_mat[int(data[vals_ind[2]] + 1), 33] += 1

    # Adding Q_5i pieces. Depending on adjacency of 3 empty voxels (adj3n), we add to 51, 52, or 53
    if adj3n >= 4.2:
        blank_mat[int(data[vals_ind[2]] + 1), 30] += 1
        blank_mat[int(data[vals_ind[3]] + 1), 31] += 1
    elif adj3n >= 4:
        blank_mat[int(data[vals_ind[2]] + 1), 28] += 1
        blank_mat[int(data[vals_ind[3]] + 1), 29] += 1
    else:
        blank_mat[int(data[vals_ind[2]] + 1), 26] += 1
        blank_mat[int(data[vals_ind[3]] + 1), 27] += 1

    # Adding Q_4i pieces. Depending on adjacency of 4 filled voxels (adj4), we add to 41, 42, 43, 44, 45, or 46
    if adj4 >= 8.4:
        blank_mat[int(data[vals_ind[3]] + 1), 24] += 1
        blank_mat[int(data[vals_ind[4]] + 1), 25] += 1
    elif adj4 >= 8.2:
        blank_mat[int(data[vals_ind[3]] + 1), 22] += 1
        blank_mat[int(data[vals_ind[4]] + 1), 23] += 1
    elif adj4 >= 7.7:
        blank_mat[int(data[vals_ind[3]] + 1), 20] += 1
        blank_mat[int(data[vals_ind[4]] + 1), 21] += 1
    elif adj4 >= 7.4:
        blank_mat[int(data[vals_ind[3]] + 1), 18] += 1
        blank_mat[int(data[vals_ind[4]] + 1), 19] += 1
    elif adj4 >= 7.0:
        blank_mat[int(data[vals_ind[3]] + 1), 16] += 1
        blank_mat[int(data[vals_ind[4]] + 1), 17] += 1
    else:
        blank_mat[int(data[vals_ind[3]] + 1), 14] += 1
        blank_mat[int(data[vals_ind[4]] + 1), 15] += 1

    # Adding Q_3i pieces. Depending on adjacency of 3 filled voxels (adj3), we add to 31, 32, or 33
    if adj3 >= 4.2:
        blank_mat[int(data[vals_ind[4]] + 1), 12] += 1
        blank_mat[int(data[vals_ind[5]] + 1), 13] += 1
    elif adj3 >= 4:
        blank_mat[int(data[vals_ind[4]] + 1), 10] += 1
        blank_mat[int(data[vals_ind[5]] + 1), 11] += 1
    else:
        blank_mat[int(data[vals_ind[4]] + 1), 8] += 1
        blank_mat[int(data[vals_ind[5]] + 1), 9] += 1

    # Adding Q_2i pieces. Depending on adjacency of 2 filled voxels (adj2), we add to 21, 22, or 23
    if adj2 >= 1.5:
        blank_mat[int(data[vals_ind[5]] + 1), 6] += 1
        blank_mat[int(data[vals_ind[6]] + 1), 7] += 1
    elif adj2 >= 1.2:
        blank_mat[int(data[vals_ind[5]] + 1), 4] += 1
        blank_mat[int(data[vals_ind[6]] + 1), 5] += 1
    else:
        blank_mat[int(data[vals_ind[5]] + 1), 2] += 1
        blank_mat[int(data[vals_ind[6]] + 1), 3] += 1

    # Adding Q_11 piece. Only one voxel active
    blank_mat[int(data[vals_ind[6]] + 1), 0] += 1
    blank_mat[int(data[vals_ind[7]] + 1), 1] += 1

    return blank_mat


# Both above methods are for superlevel set definition.
# Below are two sublevel set definitions
def compute_local_EC_2D_sublevel(data, bq_map, locs, max_val):
    vals_ind = np.argsort(data)
    # vals_ind = sorted(range(len(data)), key=data.__getitem__)
    blank_mat = bq_map

    corner = False
    edge = False
    # if data[vals_ind[1]] == (max_val + 1):
    if data[vals_ind[2]] == (-1):
        # print('Corner')
        corner = True
    # elif data[vals_ind[2]] == (max_val + 1):
    elif data[vals_ind[1]] == (-1):
        edge = True
        # print('Edge')

    # Compute adjacency of 2-pixel on configuration
    # adj = (np.linalg.norm(np.array(locs[vals_ind[0]]) - np.array(locs[vals_ind[1]])) == 1)

    adj = round(np.abs((locs[vals_ind[0]][0] - locs[vals_ind[1]][0])) +
                np.abs((locs[vals_ind[0]][1] - locs[vals_ind[1]][1]))) == 1

    if corner or edge:
        adj = 1

    # Sublevel set is reversed. Starts with q1 (single pixel). Thus, the ordering
    # is changed. Instead of starting full, it starts empty.

    # Fully filled (four pixels filled) starts at D and does not end.
    if corner or edge:
        blank_mat[-1, 8] += 1
    else:
        blank_mat[int(data[vals_ind[3]]), 8] += 1
    # blank_mat[-1, 9] += 1

    # Three pixels filled starts at C and ends at D.
    if corner or edge:
        blank_mat[-1, 6] += 1
        blank_mat[-1, 7] += 1
    else:
        blank_mat[int(data[vals_ind[2]]), 6] += 1
        blank_mat[int(data[vals_ind[3]]), 7] += 1
    # q_2 if adjacent, q_d if not
    # Two pixels filled starts at B and ends at C.
    if corner:
        blank_mat[-1, 2] += 1
        blank_mat[-1, 3] += 1
    elif edge:
        blank_mat[int(data[vals_ind[3]]), 2] += 1
        blank_mat[-1, 3] += 1
    elif adj:
        blank_mat[int(data[vals_ind[1]]), 2] += 1
        blank_mat[int(data[vals_ind[2]]), 3] += 1
    else:
        blank_mat[int(data[vals_ind[1]]), 4] += 1
        blank_mat[int(data[vals_ind[2]]), 5] += 1

    # Single pixel starts at A and ends at B.
    if corner:
        blank_mat[int(data[vals_ind[3]]), 0] += 1
        blank_mat[-1, 1] += 1
    elif edge:
        blank_mat[int(data[vals_ind[2]]), 0] += 1
        blank_mat[int(data[vals_ind[3]]), 1] += 1
    else:
        blank_mat[int(data[vals_ind[0]]), 0] += 1
        blank_mat[int(data[vals_ind[1]]), 1] += 1

    return blank_mat


def compute_local_EC_3D_sublevel(data, bq_map, locs, max_val):
    # TODO: Need to change adjacency computation as the filled in voxels are in an opposite direction!!!
    # TODO: specify corner/edge cases separately
    vals_ind = np.argsort(data)

    blank_mat = bq_map

    sqrt1 = 1 ** 0.5
    sqrt2 = 2 ** 0.5
    sqrt3 = 3 ** 0.5

    ######################################################################
    # Find the adjacency of two active voxels for Q2i
    if data[vals_ind[1]] == (max_val + 1):
        adj2 = 1
    else:
        adj2 = np.linalg.norm(locs[vals_ind[1]] - locs[vals_ind[0]])

    # Add third voxel to "total adjacency"
    if data[vals_ind[2]] == (max_val + 1):
        adj3 = 2
    else:
        adj3 = adj2 + \
               np.linalg.norm(locs[vals_ind[2]] - locs[vals_ind[1]]) + \
               np.linalg.norm(locs[vals_ind[2]] - locs[vals_ind[0]])

    # Add fourth voxel to "total adjacency"
    if data[vals_ind[3]] == (max_val + 1):
        adj4 = 3
    else:
        adj4 = adj3 + \
               np.linalg.norm(locs[vals_ind[3]] - locs[vals_ind[2]]) + \
               np.linalg.norm(locs[vals_ind[3]] - locs[vals_ind[1]]) + \
               np.linalg.norm(locs[vals_ind[3]] - locs[vals_ind[0]])

    # Uneconomical to compute fifth adjacency, but we can determine the
    # adjacency of the unactive voxels which will still allow us to
    # know the adjacency of the 5-voxel set.
    # Compute the 2 empty voxel adjacency
    if data[vals_ind[6]] == (max_val + 1):
        adj2n = 1
    else:
        adj2n = np.linalg.norm(locs[vals_ind[6]] - locs[vals_ind[7]])

    if data[vals_ind[5]] == (max_val + 1):
        adj3n = 2
    else:
        adj3n = adj2n + \
                np.linalg.norm(locs[vals_ind[5]] - locs[vals_ind[6]]) + \
                np.linalg.norm(locs[vals_ind[5]] - locs[vals_ind[7]])
    ######################################################################

    # Adding Q_81 for when everything is filled in
    blank_mat[int(data[vals_ind[7]]), 40] += 1
    # blank_mat[int(data[vals_ind[0]]), 41] += 1

    # Adding Q_71 for when everything is filled in minus one voxel
    blank_mat[int(data[vals_ind[6]]), 38] += 1
    blank_mat[int(data[vals_ind[7]]), 39] += 1

    # Adding Q_6i pieces. Depending on adjacency of 2 empty voxels (adj2n), we add to 61, 62, or 63
    if adj2n >= 1.5:
        blank_mat[int(data[vals_ind[5]]), 36] += 1
        blank_mat[int(data[vals_ind[6]]), 37] += 1
    elif adj2n >= 1.2:
        blank_mat[int(data[vals_ind[5]]), 34] += 1
        blank_mat[int(data[vals_ind[6]]), 35] += 1
    else:
        blank_mat[int(data[vals_ind[5]]), 32] += 1
        blank_mat[int(data[vals_ind[6]]), 33] += 1

    # Adding Q_5i pieces. Depending on adjacency of 3 empty voxels (adj3n), we add to 51, 52, or 53
    if adj3n >= 4.2:
        blank_mat[int(data[vals_ind[4]]), 30] += 1
        blank_mat[int(data[vals_ind[5]]), 31] += 1
    elif adj3n >= 4:
        blank_mat[int(data[vals_ind[4]]), 28] += 1
        blank_mat[int(data[vals_ind[5]]), 29] += 1
    else:
        blank_mat[int(data[vals_ind[4]]), 26] += 1
        blank_mat[int(data[vals_ind[5]]), 27] += 1

    # Adding Q_4i pieces. Depending on adjacency of 4 filled voxels (adj4), we add to 41, 42, 43, 44, 45, or 46
    if adj4 >= 8.4:
        blank_mat[int(data[vals_ind[3]]), 24] += 1
        blank_mat[int(data[vals_ind[4]]), 25] += 1
    elif adj4 >= 8.2:
        blank_mat[int(data[vals_ind[3]]), 22] += 1
        blank_mat[int(data[vals_ind[4]]), 23] += 1
    elif adj4 >= 7.7:
        blank_mat[int(data[vals_ind[3]]), 20] += 1
        blank_mat[int(data[vals_ind[4]]), 21] += 1
    elif adj4 >= 7.4:
        blank_mat[int(data[vals_ind[3]]), 18] += 1
        blank_mat[int(data[vals_ind[4]]), 19] += 1
    elif adj4 >= 7.0:
        blank_mat[int(data[vals_ind[3]]), 16] += 1
        blank_mat[int(data[vals_ind[4]]), 17] += 1
    else:
        blank_mat[int(data[vals_ind[3]]), 14] += 1
        blank_mat[int(data[vals_ind[4]]), 15] += 1

    # Adding Q_3i pieces. Depending on adjacency of 3 filled voxels (adj3), we add to 31, 32, or 33
    if adj3 >= 4.2:
        blank_mat[int(data[vals_ind[2]]), 12] += 1
        blank_mat[int(data[vals_ind[3]]), 13] += 1
    elif adj3 >= 4:
        blank_mat[int(data[vals_ind[2]]), 10] += 1
        blank_mat[int(data[vals_ind[3]]), 11] += 1
    else:
        blank_mat[int(data[vals_ind[2]]), 8] += 1
        blank_mat[int(data[vals_ind[3]]), 9] += 1

    # Adding Q_2i pieces. Depending on adjacency of 2 filled voxels (adj2), we add to 21, 22, or 23
    if adj2 >= 1.5:
        blank_mat[int(data[vals_ind[1]]), 6] += 1
        blank_mat[int(data[vals_ind[2]]), 7] += 1
    elif adj2 >= 1.2:
        blank_mat[int(data[vals_ind[1]]), 4] += 1
        blank_mat[int(data[vals_ind[2]]), 5] += 1
    else:
        blank_mat[int(data[vals_ind[1]]), 2] += 1
        blank_mat[int(data[vals_ind[2]]), 3] += 1

    # Adding Q_11 piece. Only one voxel active
    blank_mat[int(data[vals_ind[0]]), 0] += 1
    blank_mat[int(data[vals_ind[1]]), 1] += 1

    return blank_mat


# Function to return contributors to EC and Minkowski
# Method only for 2-D arrays
def get_bitquad_contributions_2D(data, max_val, superlevel=True):
    """
    Calculates the bitquad contributions for thresholds
    for EC and binary Minkowski functional. The closed set
    of possible bitquad maps is (q1, q2, qd, q3, and q4)

    :param data: numpy ND-array, image data to be analyzed
                 *pixels should have integer values*
    :param max_val: int, maximum threshold value
    :param superlevel: bool, True if using superlevel set,
                             False if using sublevel set
    :return: bitquad_map, numpy 2D-array, mapping for the
             compiled beginning and ending of features
             for all pixels (EC follows inclusion/exclusion)
    """
    thresh = range(max_val + 2)
    bitquad_mat = np.zeros((len(thresh), 5 * 2))
    data_dims = list(data.shape)

    # Mapping for checking distance for adjacency
    locs = [[0, 0], [1, 0], [0, 1], [1, 1]]
    # timings = []
    # timings_before = []
    for i in range(data_dims[0] + 1):
        for j in range(data_dims[1] + 1):
            # t_bb = t.time()
            data_local = np.zeros(4) - 1 # + (1 - superlevel) * int(max_val + 2)

            # If at a corner or edge, pixel value remains zero
            if i != 0:
                if j != 0:
                    data_local[0] = data[i - 1, j - 1]
                if j != data_dims[1]:
                    data_local[2] = data[i - 1, j]

            if i != data_dims[0]:
                if j != 0:
                    data_local[1] = data[i, j - 1]
                if j != data_dims[1]:
                    data_local[3] = data[i, j]

            # t_b = t.time()
            if superlevel:
                bitquad_mat = compute_local_EC_2D(data_local, bitquad_mat, locs, max_val)
            else:
                bitquad_mat = compute_local_EC_2D_sublevel(data_local, bitquad_mat, locs, max_val)
            # t_a = t.time()

            # timings.append(t_a - t_b)
            # timings_before.append(t_b - t_bb)

    # timings = np.asarray(timings)
    # print(np.mean(timings_before))
    # print(np.mean(timings))

    return bitquad_mat[:-1, :]


# Function to return contributors to EC and Minkowski
# Method only for 3-D arrays
def get_bitquad_contributions_3D(data, max_val, superlevel=True):
    """
    Calculates the bitquad contributions for thresholds
    for EC and binary Minkowski functional.

    :param data: numpy ND-array, image data to be analyzed
                 *pixels should have integer values*
    :param max_val: int, maximum threshold value
    :param superlevel: bool, True if using superlevel set,
                             False if using sublevel set
    :return: bitquad_map, numpy 2D-array, mapping for the
             compiled beginning and ending of features
             for all pixels (EC follows inclusion/exclusion)
    """
    thresh = range(max_val + 2)
    bitquad_mat = np.zeros((len(thresh), 21 * 2))
    data_dims = list(data.shape)

    # Mapping for checking distance for adjacency
    locs = [np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 1]),
            np.array([0, 1, 1]),
            np.array([1, 1, 1])]

    # timings = []
    # timings_before = []
    for i in range(data_dims[0] + 1):
        for j in range(data_dims[1] + 1):
            for k in range(data_dims[2] + 1):
                # t_bb = t.time()
                data_local = np.zeros(8) - 1 + (1 - superlevel) * int(max_val + 2)

                # If pixel is an edge or corner, add zeros around the data structure
                # 3D Attempt
                if i != 0:
                    if j != 0:
                        if k != 0:
                            data_local[0] = data[i - 1, j - 1, k - 1]
                        if k != data_dims[2]:
                            data_local[4] = data[i - 1, j - 1, k]
                    if j != data_dims[1]:
                        if k != 0:
                            data_local[2] = data[i - 1, j, k - 1]
                        if k != data_dims[2]:
                            data_local[6] = data[i - 1, j, k]

                if i != data_dims[0]:
                    if j != 0:
                        if k != 0:
                            data_local[1] = data[i, j - 1, k - 1]
                        if k != data_dims[2]:
                            data_local[5] = data[i, j - 1, k]
                    if j != data_dims[1]:
                        if k != 0:
                            data_local[3] = data[i, j, k - 1]
                        if k != data_dims[2]:
                            data_local[7] = data[i, j, k]

                # t_b = t.time()
                if superlevel:
                    bitquad_mat = compute_local_EC_3D(data_local, bitquad_mat, locs, max_val)
                else:
                    bitquad_mat = compute_local_EC_3D_sublevel(data_local, bitquad_mat, locs, max_val)
                # t_a = t.time()

                # timings.append(t_a - t_b)
                # timings_before.append(t_b - t_bb)

    # timings = np.asarray(timings)
    # print(np.mean(timings_before))
    # print(np.mean(timings))

    return bitquad_mat[:-1, :]


def compute_EC_curve_2D(bq_map, conn_type='8C'):
    """
    Computes the EC curve for a 2-D bitmap matrix
    found from threshold values.

    :param bq_map: numpy 2D-array, mapping for the compiled
           beginning and ending of features for pixels
    :return: EC_curve: numpy array, values for the EC for
             the integer thresholds provided by max_val
             while the contributions where found
    """
    num_thresh = bq_map.shape[0]

    if conn_type == '8C':
        contribs = np.array([0.25, -0.25, 0, 0, -0.5, 0.5, -0.25, 0.25, 0, 0])

    # This contribs multiplier treats diagonal bitquad as two connected
    # components. Gudhi uses this assumption
    if conn_type == '4C':
        contribs = np.array([0.25, -0.25, 0, 0, 0.5, -0.5, -0.25, 0.25, 0, 0])

    # Multiplying the weight vector by the bitquad_map
    # Bitquad map is of the form (N_intervals x (N_bq_types * 2))
    # contribs is of the form (1 x (N_bq_types * 2))
    EC_curve = contribs @ bq_map.T

    return EC_curve


def compute_EC_curve_3D(bq_map, conn_type='6C'):
    """
    Computes the EC curve for a 2-D bitmap matrix
    found from threshold values.

    :param bq_map: numpy 2D-array, mapping for the compiled
           beginning and ending of features for pixels
    :return: EC_curve: numpy array, values for the EC for
             the integer thresholds provided by max_val
             while the contributions where found
    """

    # This contribs multiplier is considering 6-connectedness
    # i.e., only face-adjacency is considered "adjacent"
    # Another way to look at this is that empty voxels are
    # vertex adjacent (empty voxels are 26-connected)
    if conn_type == '6C':
        contribs = np.array([0.125, -0.125,   # Q11
                             0, 0,            # Q21
                             0.25, -0.25,     # Q22
                             0.25, -0.25,     # Q23
                             -0.125, 0.125,   # Q31
                             0.125, -0.125,   # Q32
                             0.375, -0.375,   # Q33
                             0, 0,            # Q41
                             -0.25, 0.25,     # Q42
                             -0.25, 0.25,     # Q43
                             0, 0,            # Q44
                             0, 0,            # Q45
                             0.5, -0.5,       # Q46
                             -0.125, 0.125,   # Q51
                             -0.375, 0.375,   # Q52
                             -0.125, 0.125,   # Q53
                             0, 0,            # Q61
                             -0.25, 0.25,     # Q62
                             -0.75, 0.75,     # Q63
                             0.125, -0.125,   # Q71
                             0, 0])           # Q81

    # This contribs multiplier is considering 18-connectedness
    # i.e., now only edge-adjacency is considered "adjacent"
    if conn_type == '18C':
        contribs = np.array([0.125, -0.125,  # Q11
                             0, 0,  # Q21
                             -0.25, 0.25,  # Q22
                             0.25, -0.25,  # Q23
                             -0.125, 0.125,  # Q31
                             -0.375, 0.375,  # Q32
                             -0.125, 0.125,  # Q33
                             0, 0,  # Q41
                             -0.25, 0.25,  # Q42
                             -0.25, 0.25,  # Q43
                             0, 0,  # Q44
                             0, 0,  # Q45
                             0.5, -0.5,  # Q46
                             -0.125, 0.125,  # Q51
                             0.125, -0.125,  # Q52
                             0.375, -0.375,  # Q53
                             0, 0,  # Q61
                             0.25, -0.25,  # Q62
                             0.75, -0.75,  # Q63
                             0.125, -0.125,  # Q71
                             0, 0])  # Q81

    # This contribs multiplier is considering 26-connectedness
    # i.e., now only edge-adjacency is considered "adjacent"
    if conn_type == '26C':
        contribs = np.array([0.125, -0.125,  # Q11
                             0, 0,  # Q21
                             -0.25, 0.25,  # Q22
                             -0.75, 0.75,  # Q23
                             -0.125, 0.125,  # Q31
                             -0.375, 0.375,  # Q32
                             -0.125, 0.125,  # Q33
                             0, 0,  # Q41
                             -0.25, 0.25,  # Q42
                             -0.25, 0.25,  # Q43
                             0, 0,  # Q44
                             0, 0,  # Q45
                             0.5, -0.5,  # Q46
                             -0.125, 0.125,  # Q51
                             0.125, -0.125,  # Q52
                             0.375, -0.375,  # Q53
                             0, 0,  # Q61
                             0.25, -0.25,  # Q62
                             0.25, -0.25,  # Q63
                             0.125, -0.125,  # Q71
                             0, 0])  # Q81

    # This contribs multiplier is considering 18-connectedness complement
    # i.e., now only edge-adjacency is considered "adjacent"
    if conn_type == '18\'C':
        contribs = np.array([0.125, -0.125,   # Q11
                             0, 0,            # Q21
                             0.25, -0.25,     # Q22
                             0.25, -0.25,     # Q23
                             -0.125, 0.125,   # Q31
                             0.125, -0.125,   # Q32
                             0.375, -0.375,   # Q33
                             0, 0,            # Q41
                             -0.25, 0.25,     # Q42
                             -0.25, 0.25,     # Q43
                             0, 0,            # Q44
                             0, 0,            # Q45
                             0.5, -0.5,       # Q46
                             -0.125, 0.125,   # Q51
                             -0.375, 0.375,   # Q52
                             -0.125, 0.125,   # Q53
                             0, 0,            # Q61
                             -0.25, 0.25,     # Q62
                             0.25, -0.25,     # Q63
                             0.125, -0.125,   # Q71
                             0, 0])           # Q81

    # Multiplying the weight vector by the bitquad_map
    # Bitquad map is of the form (N_intervals x (N_bq_types * 2))
    # contribs is of the form (1 x (N_bq_types * 2))
    EC_curve = contribs @ bq_map.T

    return EC_curve


def get_data_attr(file_path):
    """
    Extract HSI file dimensions from header file

    :param file_path: path or string, path to header file
    :return: list (len=3), dimensions of HSI [x, y, lambda]
    """
    dimensions = [0, 0, 0]

    with open(file_path) as f:
        ripe_hdr = f.readlines()

        for i in ripe_hdr:
            items = i.split()

            if items[0] == 'samples':
                dimensions[1] = int(items[2])
            elif items[0] == 'lines':
                dimensions[0] = int(items[2])
            elif items[0] == 'bands':
                dimensions[2] = int(items[2])

    return dimensions
