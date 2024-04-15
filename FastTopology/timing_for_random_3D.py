import numpy as np
from py_calls_to_cpp import call_cpp_3D, call_cpp_3D_parallel
import time as t
import csv

import sys

from EC_computation_funcs import gudhi_get_EC_3D, compute_EC_curve_3D


def rescale_array(data=None, max_val=255):
    A = np.min(data)
    B = np.max(data)

    scaled_data = ((data - A) / (B - A) * 255).astype(np.intc)
    return scaled_data


# Recording everything:
#               GUDHI
#               Serial
#               2 cores
#               3 cores
#               4 cores
cols = ['GUDHI_u', 'GUDHI_n', 'Serial_u', 'Serial_n',
        'P2_u', 'P2_n', 'P3_u', 'P3_n', 'P4_u', 'P4_n',
        'P5_u', 'P5_n', 'P6_u', 'P6_n', 'P7_u', 'P7_n',
        'P8_u', 'P8_n', 'P12_u', 'P12_n', 'P16_u', 'P16_n',
        'P20_u', 'P20_n', 'P24_u', 'P24_n', 'P32_u', 'P32_n']


width_in = int(sys.argv[1])
height_in = int(sys.argv[2])
depth_in = int(sys.argv[3])
num_samples = int(sys.argv[4])
filename = sys.argv[5]

print(num_samples, width_in, height_in, depth_in)

num_samples = 1


# Timing for intermediate results
t00 = t.time()

# Write first row
with open(filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(cols)

for i in range(num_samples):
    my_vals = []
    rand_u = np.random.randint(0, 255, (depth_in, width_in, height_in)).astype(np.intc)
    rand_n = rescale_array(np.random.normal(0, 1, (depth_in, width_in, height_in))).astype(np.intc)

    width = depth_in
    height = width_in
    depth = height_in

    # GUDHI timings
    t_0 = t.time()
    gudhi_get_EC_3D(rand_u / 255.0, 256, 0, 1)
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    gudhi_get_EC_3D(rand_n / 255.0, 256, 0, 1)
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Serial timings
    t_0 = t.time()
    contr = call_cpp_3D(rand_u, width, height, depth, 255, 0)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D(rand_n, width, height, depth, 255, 0)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 2-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 2)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 2)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 3-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 3)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 3)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 4-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 4)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 4)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 5-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 5)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 5)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 6-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 6)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 6)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 7-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 7)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 7)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 8-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 8)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 8)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 12-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 12)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 12)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 16-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 16)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 16)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 20-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 20)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 20)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 24-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 24)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 24)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 32-core
    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_u, width, height, depth, 255, 32)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 32)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(my_vals)

    if (i % 50) == 0:
        t11 = t.time()
        print('50 more runs done: {:.2f} seconds total'.format(t11 - t00))