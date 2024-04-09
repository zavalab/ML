//
//  fast_EC_functions.hpp
//  fast_EC
//
//  Created by Daniel Laky on 7/3/23.
//

#ifndef fast_EC_functions_hpp
#define fast_EC_functions_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>

float pixel_adjacency(int pos1[], int pos2[], int n_dims);

void hex_decoder(char hex_char, int dec_rep);

int decode_little_endian_BMP(char *input_hexstring, int str_len, bool signed_int=false);

int decode_little_endian_BMP_int(int *input_int, int str_len, bool signed_int=false);

void argsort(int data[], int locs[], int n_dims);

void local_contr_2D_sub(int data[], int bitmap[][10]);

void local_contr_2D_sub_flat(int data[], int bitmap[]);

void local_contr_2D_sub_flat_spl(int data[], int bitmap[]);

void local_contr_2D_sub_flat_mtx(int data[], int bitmap[]);

void copy_points(int locs[8][3], int temp_loc[3], int ind);

void local_contr_3D_sub(int data[], int bitmap[][42]);

void local_contr_3D_sub_flat(int data[], int bitmap[]);

extern "C" void get_unit_contributions_2D(int *data, int contr_map[][10], int dim_x, int dim_y, int max_val, int size_data, int data_start_index, bool sup_level);

extern "C" void get_unit_contr_2D_flattened(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int size_data, int data_start_index, bool sup_level);

void compute_contr_parallel_CPU(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int data_start_index, int num_threads);

extern "C" void compute_contr_parallel_CPU_comb(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int data_start_index, int num_threads);

void compute_contr_parallel_GPU(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int data_start_index);

extern "C" void get_unit_contr_3D_flattened(int *data, int* contr_map, int dim_x, int dim_y, int dim_z, int max_val, int size_data, int data_start_index, bool sup_level);

extern "C" void compute_contr_parallel_CPU_comb_3D(int *data, int *contr_map, int dim_x, int dim_y, int dim_z, int max_val, int data_start_index, int num_threads);

extern "C" void compute_contr_2D_low_mem(char *filename, int *contr_map, int max_val, long int start_val, long int total_jobs);

extern "C" void compute_contr_2D_low_mem_parallel(char *filename, int *contr_map, int max_val, int num_threads);

#endif /* fast_EC_functions_hpp */
