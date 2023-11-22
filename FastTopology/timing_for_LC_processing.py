import json
import glob
import re
import numpy as np
import cv2
from py_calls_to_cpp import call_cpp_2D, call_cpp_2D_parallel
import time as t
import csv
import os.path

import sys

from EC_computation_funcs import gudhi_get_EC_2D, compute_EC_curve_2D


def rescale_array(data=None, max_val=255):
    A = np.min(data)
    B = np.max(data)

    scaled_data = ((data - A) / (B - A) * 255).astype(np.intc)
    return scaled_data


# User supplied filename for saving timing results
filename = sys.argv[1]

# File path definitions
files = glob.glob('data/LC/*.png')

cols = ['GUDHI', 'GUDHI_n', 'Serial', 'Serial_n',
        'P2', 'P2_n', 'P3', 'P3_n', 'P4', 'P4_n',
        'P5', 'P5_n', 'P6', 'P6_n', 'P7', 'P7_n',
        'P8', 'P8_n', 'P12', 'P12_n', 'P16', 'P16_n',
        'P20', 'P20_n', 'P24', 'P24_n', 'P32', 'P32_n']

# Timing for intermediate results
t00 = t.time()

# Write first row
with open(filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(cols)

count = 0

for i in files:
    my_vals = []

    im = cv2.imread(os.path.abspath(i))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    raw_data = im[:, :]

    width = im.shape[0]
    height = im.shape[1]
    rand_n = rescale_array(np.random.normal(0, 1, (width * height))).astype(np.intc)

    raw_data = rescale_array(raw_data, max_val=255).astype(np.intc)

    # GUDHI timings
    t_0 = t.time()
    thing = gudhi_get_EC_2D(np.reshape(raw_data, (width, height)).astype(np.float32) / 255.0, 256, 0, 1)
    t_1 = t.time()
    my_vals.append(t_1 - t_0)


    t_0 = t.time()
    thing = gudhi_get_EC_2D(np.reshape(rand_n, (width, height)).astype(np.float32) / 255.0, 256, 0, 1)
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Serial timings
    t_0 = t.time()
    contr = call_cpp_2D(raw_data, width, height, 255, 0)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)


    t_0 = t.time()
    contr = call_cpp_2D(rand_n, width, height, 255, 0)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 2-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 2)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)


    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 2)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 3-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 3)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 3)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 4-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 4)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 4)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 5-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 5)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 5)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 6-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 6)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 6)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 7-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 7)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 7)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 8-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 8)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 8)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 12-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 12)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 12)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 16-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 16)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 16)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 20-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 20)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 20)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 24-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 24)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 24)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Parallel timings; 32-core
    t_0 = t.time()
    contr = call_cpp_2D_parallel(raw_data, width, height, 255, 32)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_2D_parallel(rand_n, width, height, 255, 32)
    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    # Save the data
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(my_vals)

    if (count % 50) == 0:
        t11 = t.time()
        print('50 more runs done: {:.2f} seconds total'.format(t11 - t00))

    count += 1
