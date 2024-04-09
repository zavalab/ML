import json
import glob
import re
import numpy as np
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


def id2fruit(ind, fruit_ids):
    """
    Function to take an index and return which
    fruit is being analyzed from the data set.

    :param ind: int, which index is being analyzed
    :param fruit_ids: dict, keys    --> fruits (str)
                            values  --> list of indices
                                        for each fruit
    :return: fruit, string of fruit name for index
    """
    fruit = None

    for i in fruit_ids.keys():
        if ind in fruit_ids[i]:
            fruit = i
            break

    return fruit


def find_int(name):
    return int(re.search(r'\d+', name).group())


###########################################################
# Steps of this code
###########################################################
# 1. Identify which indices are which fruits
# 2. Gather all indices that are annotated which are Kiwi
# 3. Read data for a Kiwi HSI file (with NIR camera)
#       a. Use header file to find dimensions
#       b. Read data file using numpy
#       c. Cast numpy data to correct size
# 4. Use raw data to find EC curve
#       a. Perform EC using GUDHI
#       b. Perform EC using cpp code with varying cores
# 5. Save data
#       a. Append values to specified csv file
###########################################################

# Data path for our data
raw_data_path = 'data/HSI/'
fruit_examined = 'Kiwi'

# Identify which indices are which fruits
with open('data/HSI/train_all_v2.json') as f:
    train_all = json.load(f)

fruit_ids = {'Avocado': [],
             'Mango': [],
             'Kaki': [],
             'Kiwi': [],
             'Papaya': []}

for ind, record in enumerate(train_all['records']):
    if record['fruit'] in fruit_ids.keys():
        fruit_ids[record['fruit']].append(record['id'])

# Gather which indices are annotated as our fruit
labelled_fruits_id = []
labelled_fruits_recordid = []

for ind, val in enumerate(train_all['annotations']):
    current_fruit = id2fruit(val['record_id'], fruit_ids=fruit_ids)

    if current_fruit == fruit_examined:
        labelled_fruits_recordid.append(val['record_id'])
        labelled_fruits_id.append(ind)

cols = ['GUDHI', 'GUDHI_n', 'Serial', 'Serial_n',
        'P2', 'P2_n', 'P3', 'P3_n', 'P4', 'P4_n',
        'P5', 'P5_n', 'P6', 'P6_n', 'P7', 'P7_n',
        'P8', 'P8_n', 'P12', 'P12_n', 'P16', 'P16_n',
        'P20', 'P20_n', 'P24', 'P24_n', 'P32', 'P32_n']

filename = sys.argv[1]

# Write first row
with open(filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(cols)

t00 = t.time()
count = 0
for j in range(len(fruit_ids[fruit_examined])):
    my_vals = []

    i = fruit_ids[fruit_examined][j]
    camera = train_all['records'][i]['camera_type']

    hdr_file = raw_data_path + train_all['records'][i]['files']['header_file']
    bin_file = raw_data_path + train_all['records'][i]['files']['data_file']

    if not os.path.exists(hdr_file):
        continue

    # Read data
    dims = get_data_attr(hdr_file)
    raw_data = np.fromfile(bin_file, dtype=np.float32)

    width = dims[0]
    height = dims[1]
    depth = dims[2]
    rand_n = rescale_array(np.random.normal(0, 1, (width*height*depth))).astype(np.intc)

    raw_data[raw_data > 1] = 1
    raw_data[raw_data < 0] = 0

    raw_data = rescale_array(raw_data, max_val=255).astype(np.intc)

    # GUDHI timings
    t_0 = t.time()
    thing = gudhi_get_EC_3D(np.reshape(raw_data, (dims[2], dims[0], dims[1])).astype(np.float32) / 255.0, 256, 0, 1)
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    thing = gudhi_get_EC_3D(np.reshape(rand_n, (dims[2], dims[0], dims[1])).astype(np.float32) / 255.0, 256, 0, 1)
    t_1 = t.time()
    my_vals.append(t_1 - t_0)


    # Serial timings
    t_0 = t.time()
    contr = call_cpp_3D(raw_data, width, height, depth, 255, 0)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 2)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 3)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 4)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 5)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 6)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 7)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 8)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 12)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 16)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 20)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 24)
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
    contr = call_cpp_3D_parallel(raw_data, width, height, depth, 255, 32)
    ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
    t_1 = t.time()
    my_vals.append(t_1 - t_0)

    t_0 = t.time()
    contr = call_cpp_3D_parallel(rand_n, width, height, depth, 255, 32)
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
