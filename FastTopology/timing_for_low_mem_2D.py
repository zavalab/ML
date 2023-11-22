import numpy as np
from py_calls_to_cpp import call_cpp_2D_low_memory, call_cpp_2D_low_memory_parallel
import time as t
import csv
from PIL import Image

import sys

from EC_computation_funcs import compute_EC_curve_2D, gudhi_get_EC_2D


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
cols = ['Serial_u', 'Serial_n',
        'P2_u', 'P2_n', 'P3_u', 'P3_n', 'P4_u', 'P4_n',
        'P5_u', 'P5_n', 'P6_u', 'P6_n', 'P7_u', 'P7_n',
        'P8_u', 'P8_n', 'P12_u', 'P12_n', 'P16_u', 'P16_n',
        'P20_u', 'P20_n', 'P24_u', 'P24_n']

width = int(sys.argv[1])
height = int(sys.argv[2])
num_samples = int(sys.argv[3])
filename = sys.argv[4]

# Timing for intermediate results
t00 = t.time()

# Write first row
with open(filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(cols)

for i in range(num_samples):
    my_vals = []
    rand_u = np.random.randint(0, 255, (width, height)).astype(np.uint8)
    rand_n = rescale_array(np.random.normal(0, 1, (width, height))).astype(np.uint8)

    # Format properly for PIL [numpy has 2, 0, 1 ordering]
    img_array = np.array([rand_u, rand_u, rand_u]).transpose(1, 2, 0)
    img_array2 = np.array([rand_n, rand_n, rand_n]).transpose(1, 2, 0)

    unifrom_img = Image.fromarray(img_array)
    unifrom_img.save('out_uniform.bmp')

    normal_img = Image.fromarray(img_array2)
    normal_img.save('out_normal.bmp')

    rand_n = 'out_normal.bmp'
    rand_u = 'out_uniform.bmp'

    # Serial timings
    t_0 = t.time()
    contr = call_cpp_2D_low_memory(rand_u, 255, 0)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory(rand_n, 255, 0)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 2-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 2)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 2)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 3-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 3)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 3)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 4-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 4)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 4)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 5-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 5)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 5)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 6-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 6)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 6)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 7-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 7)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 7)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 8-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 8)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 8)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 12-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 12)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 12)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 16-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 16)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 16)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 20-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 20)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 20)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 24-core
    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_u, 255, 0, 24)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_low_memory_parallel(rand_n, 255, 0, 24)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(my_vals)

    if (i % 50) == 0:
        t11 = t.time()
        print('50 more runs done: {:.2f} seconds total'.format(t11 - t00))
