import json
import glob
import re
import numpy as np
import cv2
from py_calls_to_cpp import call_cpp_3D, call_cpp_3D_parallel
import time as t
import csv
import os.path

import sys

from EC_computation_funcs import gudhi_get_EC_3D, compute_EC_curve_3D


def rescale_array(data=None, max_val=255):
    A = np.min(data)
    B = np.max(data)

    scaled_data = ((data - A) / (B - A) * 255).astype(np.intc)
    return scaled_data


# User supplied filename for saving timing results
filename = sys.argv[1]

# 3D data
instances = np.load("data/MD_data/MD_Training_Sets.pickle",
                    allow_pickle=True)

samples = len(instances)

x_array = np.asarray(instances[0])
y_array = np.asarray(instances[1])
############################################################
cols = ['GUDHI', 'Serial', 'P2', 'P3', 'P4', 'P5',
        'P6', 'P7', 'P8', 'P12', 'P16', 'P20', 'P24']

# Timing for intermediate results
t00 = t.time()

# Write first row
with open(filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(cols)

count = 0

for i in range(samples):
    for j in range(10):
        my_vals = []

        raw_data = x_array[i, j, :, :, :, 0]

        width, height, depth = raw_data.shape

        raw_data = rescale_array(raw_data, max_val=255).astype(np.intc)

        # GUDHI timings
        t_0 = t.time()
        thing = gudhi_get_EC_3D(np.reshape(raw_data, (width, height, depth)).astype(np.float32) / 255.0, 256, 0, 1)
        t_1 = t.time()
        my_vals.append(t_1 - t_0)

        # Serial timings
        t_0 = t.time()
        contr = call_cpp_3D(raw_data, width, height, depth, 255, 0)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)

        # Parallel timings; 2-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 2)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)

        # Parallel timings; 3-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 3)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)


        # Parallel timings; 4-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 4)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)


        # Parallel timings; 5-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 5)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)


        # Parallel timings; 6-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 6)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)


        # Parallel timings; 7-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 7)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)


        # Parallel timings; 8-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 8)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)


        # Parallel timings; 12-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 12)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)


        # Parallel timings; 16-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 16)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)

        # Parallel timings; 20-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 20)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
        t_1 = t.time()
        my_vals.append(t_1 - t_0)

        # Parallel timings; 24-core
        t_0 = t.time()
        contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 24)
        ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
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
