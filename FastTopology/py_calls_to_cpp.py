import ctypes
import numpy as np

# Library from cpp
lib = ctypes.CDLL("./fast_EC_functions.so")


def call_cpp_2D(data=None, dim_x=None, dim_y=None, max_val=255, start_index=0):
    """
    Function to call cpp to compute bitmap contributions for a 2D image using
    serial computation.

    :param data: np.array, 2D array containing image data in c-integer format (if not, it will be cast)
    :param dim_x: int, width of the image data
    :param dim_y: int, height of the image data
    :param max_val: int, maximum value of image data (typically data is [0, 256))
    :param start_index: int, where to start on the data. Default value of 0 [the whole image]
    :return: contr_map, 2D np.array [shape (max_val + 2, 10)] of the contributions of the bitquads
                        over the filtration map values from 0 to max_val
    """
    get_contr = lib.get_unit_contr_2D_flattened
    get_contr.argtypes = [ctypes.POINTER(ctypes.c_int),  # Data
                          ctypes.POINTER(ctypes.c_int),  # Contribution array
                          ctypes.c_int,  # dim_x
                          ctypes.c_int,  # dim_y
                          ctypes.c_int,  # max_val
                          ctypes.c_int,  # size_data (dim_x + 1) * (dim_y + 1)
                          ctypes.c_int,  # start index (which vertex we start the algorithm at)
                          ctypes.c_bool]  # Whether to do sup or not --> NOT IMPLEMENTED, USE FALSE

    # Ensure data is above 0
    assert np.all(data >= 0), "Ensure that your data is cast to integer and is positive. Translate data if necessary."
    # Ensure all data is below 256
    assert np.all(data <= max_val), "All values of your image must be below the max_val given to the function."
    # Check data type of the image data; must be np.intc
    assert data.dtype == np.int32, "Data type of image data must be np.intc; please cast your data before passing " \
                                 "to this function!"

    # Establish contribution map
    try:
        contr_map = np.zeros((max_val + 2, 10), dtype=np.intc).flatten()
    except:
        ValueError("max_val keyword should be an integer value. Typically, we scale image data to uint8 [0, 256)")

    data = data.flatten()

    # Create c_types for data and contribution inputs
    data_arr_c = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    contr_map_c = contr_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Create c_types for other inputs
    dim_x_c = ctypes.c_int(dim_x)
    dim_y_c = ctypes.c_int(dim_y)
    max_val_c = ctypes.c_int(max_val)
    data_start_c = ctypes.c_int(start_index)

    # Create c_types that are not input
    size_data_c = ctypes.c_int((dim_x + 1) * (dim_y + 1))
    sup_bool_c = ctypes.c_bool(False)

    # Call the function!
    get_contr(data_arr_c, contr_map_c, dim_x_c, dim_y_c, max_val_c, size_data_c, data_start_c, sup_bool_c)

    # Get output and reshape
    for i in range((max_val + 2) * 10):
        contr_map[i] = contr_map_c[i]
    contr_map = np.reshape(contr_map, (max_val + 2, 10))

    return contr_map


def call_cpp_2D_parallel(data=None, dim_x=None, dim_y=None, max_val=255, num_threads=2):
    """
        Function to call cpp to compute bitmap contributions for a 2D image using
        parallel computation.

        :param data: np.array, 2D array containing image data in c-integer format (if not, it will be cast)
        :param dim_x: int, width of the image data
        :param dim_y: int, height of the image data
        :param max_val: int, maximum value of image data (typically data is [0, 256))
        :param num_threads: int, number of cores to use during computation
        :return: contr_map, 2D np.array [shape (max_val + 2, 10)] of the contributions of the bitquads
                            over the filtration map values from 0 to max_val
        """
    get_contr = lib.compute_contr_parallel_CPU_comb
    get_contr.argtypes = [ctypes.POINTER(ctypes.c_int),  # Data
                          ctypes.POINTER(ctypes.c_int),  # Contribution array
                          ctypes.c_int,  # dim_x
                          ctypes.c_int,  # dim_y
                          ctypes.c_int,  # max_val
                          ctypes.c_int,  # size_data (dim_x + 1) * (dim_y + 1)
                          ctypes.c_int]  # start index (which vertex we start the algorithm at)

    # Ensure data is above 0
    assert np.all(data >= 0), "Ensure that your data is cast to integer and is positive. Translate data if necessary."
    # Ensure all data is below 256
    assert np.all(data <= max_val), "All values of your image must be below the max_val given to the function."
    # Check data type of the image data; must be np.intc
    assert data.dtype == np.int32, "Data type of image data must be np.intc; please cast your data before passing " \
                                 "to this function!"

    # Establish contribution map
    try:
        contr_map = np.zeros((max_val + 2, 10), dtype=np.intc).flatten()
    except:
        ValueError("max_val keyword should be an integer value. Typically, we scale image data to uint8 [0, 256)")

    # Ensure number of threads is greater than 1
    assert int(num_threads) > 1, "If using the parallel function, please use more than one core. " \
                                 "Otherwise use the serial version!"

    data = data.flatten()

    # Create c_types for data and contribution inputs
    data_arr_c = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    contr_map_c = contr_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Create c_types for other inputs
    dim_x_c = ctypes.c_int(dim_x)
    dim_y_c = ctypes.c_int(dim_y)
    max_val_c = ctypes.c_int(max_val)
    num_threads_c = ctypes.c_int(num_threads)

    # Create c_types that are not input
    size_data_c = ctypes.c_int((dim_x + 1) * (dim_y + 1))

    # Call the function!
    get_contr(data_arr_c, contr_map_c, dim_x_c, dim_y_c, max_val_c, size_data_c, num_threads_c)

    # Get output and reshape
    for i in range((max_val + 2) * 10):
        contr_map[i] = contr_map_c[i]
    contr_map = np.reshape(contr_map, (max_val + 2, 10))

    return contr_map


def call_cpp_2D_low_memory(data_file=None, max_val=255, start_val=0):
    """
            Function to call cpp to compute bitmap contributions for a 2D image using
            serial computation but only using 4 pieces of the image data.

            :param data_file: string, absolute file path to image data file in BMP format
            :param max_val: int, maximum value of image data (typically data is [0, 256))
            :param start_val: int, starting index for the vertices ()
            :return: contr_map, 2D np.array [shape (max_val + 2, 10)] of the contributions of the bitquads
                                over the filtration map values from 0 to max_val
            """
    get_contr = lib.compute_contr_2D_low_mem
    get_contr.argtypes = [ctypes.POINTER(ctypes.c_char),  # Data file name
                          ctypes.POINTER(ctypes.c_int),  # Contribution map
                          ctypes.c_int,  #
                          ctypes.c_long]  #

    # Establish contribution map
    try:
        contr_map = np.zeros((max_val + 2, 10), dtype=np.intc).flatten()
    except:
        ValueError("max_val keyword should be an integer value. Typically, we scale image data to uint8 [0, 256)")

    # Create c_types for data and contribution inputs
    filename_c = data_file.encode()
    contr_map_c = contr_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Create c_types for other inputs
    max_val_c = ctypes.c_int(max_val)
    start_index_c = ctypes.c_long(start_val)
    total_jobs_c = ctypes.c_long(0)

    # Call the function!
    get_contr(filename_c, contr_map_c, max_val_c, start_index_c, total_jobs_c)

    # Get output and reshape
    for i in range((max_val + 2) * 10):
        contr_map[i] = contr_map_c[i]
    contr_map = np.reshape(contr_map, (max_val + 2, 10))

    return contr_map


def call_cpp_2D_low_memory_parallel(data_file=None, max_val=255, start_val=0, num_threads=2):
    """
            Function to call cpp to compute bitmap contributions for a 2D image using
            serial computation but only using 4 pieces of the image data.

            :param data_file: string, absolute file path to image data file in BMP format
            :param max_val: int, maximum value of image data (typically data is [0, 256))
            :param start_val: int, starting index for the vertices ()
            :param num_threads: int, number of threads
            :return: contr_map, 2D np.array [shape (max_val + 2, 10)] of the contributions of the bitquads
                                over the filtration map values from 0 to max_val
            """
    get_contr = lib.compute_contr_2D_low_mem_parallel
    get_contr.argtypes = [ctypes.POINTER(ctypes.c_char),  # Data file name
                          ctypes.POINTER(ctypes.c_int),  # Contribution map
                          ctypes.c_int,  # Max value
                          ctypes.c_int]  # Num threads

    # Establish contribution map
    try:
        contr_map = np.zeros((max_val + 2, 10), dtype=np.intc).flatten()
    except:
        ValueError("max_val keyword should be an integer value. Typically, we scale image data to uint8 [0, 256)")

    assert int(num_threads) > 1, "If using the parallel function, please use more than one core. " \
                                 "Otherwise use the serial version!"

    # Create c_types for data and contribution inputs
    filename_c = data_file.encode()
    contr_map_c = contr_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Create c_types for other inputs
    max_val_c = ctypes.c_int(max_val)
    num_threads_c = ctypes.c_int(num_threads)

    # Call the function!
    get_contr(filename_c, contr_map_c, max_val_c, num_threads_c)

    # Get output and reshape
    for i in range((max_val + 2) * 10):
        contr_map[i] = contr_map_c[i]
    contr_map = np.reshape(contr_map, (max_val + 2, 10))

    return contr_map


def call_cpp_3D(data=None, dim_x=None, dim_y=None, dim_z=None, max_val=255, start_index=0):
    """
        Function to call cpp to compute bitmap contributions for a 3D image using
        serial computation.

        :param data: np.array, 3D array containing image data in c-integer format (if not, it will be cast)
        :param dim_x: int, width of the image data
        :param dim_y: int, height of the image data
        :param dim_z: int, depth of the image data
        :param max_val: int, maximum value of image data (typically data is [0, 256))
        :param start_index: int, where to start on the data. Default value of 0 [the whole image]
        :return: contr_map, 2D np.array [shape (max_val + 2, 42)] of the contributions of the bitquads
                            over the filtration map values from 0 to max_val
        """
    get_contr = lib.get_unit_contr_3D_flattened
    get_contr.argtypes = [ctypes.POINTER(ctypes.c_int),  # Data
                          ctypes.POINTER(ctypes.c_int),  # Contribution map
                          ctypes.c_int,  # dim_x
                          ctypes.c_int,  # dim_y
                          ctypes.c_int,  # dim_z
                          ctypes.c_int,  # max_val
                          ctypes.c_int,  # size_data (dim_x + 1) * (dim_y + 1)
                          ctypes.c_int,  # start index (which vertex we start the algorithm at)
                          ctypes.c_bool]  # Whether to do sup or not --> NOT IMPLEMENTED, USE FALSE

    # Ensure data is above 0
    assert np.all(data >= 0), "Ensure that your data is cast to integer and is positive. Translate data if necessary."
    # Ensure all data is below 256
    assert np.all(data <= max_val), "All values of your image must be below the max_val given to the function."
    # Check data type of the image data; must be np.intc
    assert data.dtype == np.int32, "Data type of image data must be np.intc; please cast your data before passing " \
                                 "to this function!"

    # Establish contribution map
    try:
        contr_map = np.zeros((max_val + 2, 42), dtype=np.intc).flatten()
    except:
        ValueError("max_val keyword should be an integer value. Typically, we scale image data to uint8 [0, 256)")

    data = data.flatten()

    # Create c_types for data and contribution inputs
    data_arr_c = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    contr_map_c = contr_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Create c_types for other inputs
    dim_x_c = ctypes.c_int(dim_x)
    dim_y_c = ctypes.c_int(dim_y)
    dim_z_c = ctypes.c_int(dim_z)
    max_val_c = ctypes.c_int(max_val)
    data_start_c = ctypes.c_int(start_index)

    # Create c_types that are not input
    size_data_c = ctypes.c_int((dim_x + 1) * (dim_y + 1) * (dim_z + 1))
    sup_bool_c = ctypes.c_bool(False)

    # Call the function!
    get_contr(data_arr_c, contr_map_c, dim_x_c, dim_y_c, dim_z_c, max_val_c, size_data_c, data_start_c, sup_bool_c)

    # Get output and reshape
    for i in range((max_val + 2) * 42):
        contr_map[i] = contr_map_c[i]
    contr_map = np.reshape(contr_map, (max_val + 2, 42))

    return contr_map


def call_cpp_3D_parallel(data=None, dim_x=None, dim_y=None, dim_z=None, max_val=255, num_threads=2):
    """
            Function to call cpp to compute bitmap contributions for a 3D image using
            serial computation.

            :param data: np.array, 2D array containing image data in c-integer format (if not, it will be cast)
            :param dim_x: int, width of the image data
            :param dim_y: int, height of the image data
            :param dim_z: int, depth of the image data
            :param max_val: int, maximum value of image data (typically data is [0, 256))
            :param num_threads: int, number of cores to use for parallel implementation
            :return: contr_map, 2D np.array [shape (max_val + 2, 42)] of the contributions of the bitquads
                                over the filtration map values from 0 to max_val
            """
    get_contr = lib.compute_contr_parallel_CPU_comb_3D
    get_contr.argtypes = [ctypes.POINTER(ctypes.c_int),  # Data
                          ctypes.POINTER(ctypes.c_int),  # Contribution map
                          ctypes.c_int,  # dim_x
                          ctypes.c_int,  # dim_y
                          ctypes.c_int,  # dim_z
                          ctypes.c_int,  # max_val
                          ctypes.c_int,  # start index (which vertex we start the algorithm at)
                          ctypes.c_int]  # Number of threads

    # Ensure data is above 0
    assert np.all(data >= 0), "Ensure that your data is cast to integer and is positive. Translate data if necessary."
    # Ensure all data is below 256
    assert np.all(data <= max_val), "All values of your image must be below the max_val given to the function."
    # Check data type of the image data; must be np.intc
    assert data.dtype == np.int32, "Data type of image data must be np.intc; please cast your data before passing " \
                                 "to this function!"

    # Establish contribution map
    try:
        contr_map = np.zeros((max_val + 2, 42), dtype=np.intc).flatten()
    except:
        ValueError("max_val keyword should be an integer value. Typically, we scale image data to uint8 [0, 256)")

    # Ensure number of threads is > 1
    assert int(num_threads) > 1, "If using the parallel function, please use more than one core. " \
                                 "Otherwise use the serial version!"

    data = data.flatten()

    # Create c_types for data and contribution inputs
    data_arr_c = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    contr_map_c = contr_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Create c_types for other inputs
    dim_x_c = ctypes.c_int(dim_x)
    dim_y_c = ctypes.c_int(dim_y)
    dim_z_c = ctypes.c_int(dim_z)
    max_val_c = ctypes.c_int(max_val)
    data_start_c = ctypes.c_int(0)
    num_threads_c = ctypes.c_int(num_threads)

    # Call the function!
    get_contr(data_arr_c, contr_map_c, dim_x_c, dim_y_c, dim_z_c, max_val_c, data_start_c, num_threads_c)

    # Get output and reshape
    for i in range((max_val + 2) * 42):
        contr_map[i] = contr_map_c[i]
    contr_map = np.reshape(contr_map, (max_val + 2, 42))

    return contr_map
