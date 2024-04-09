//
//  fast_EC_functions.cpp
//  fast_EC
//
//  Created by Daniel Laky on 7/3/23.
//

#include "fast_EC_functions.hpp"

//class Spinlock {
//    private: std::atomic_flag atomic_flag = ATOMIC_FLAG_INIT;
    
//    public: void lock() {
//        for (;;) {
//            if (!atomic_flag.test_and_set(std::memory_order_acquire)) {
//                break;
//            }
//            while (atomic_flag.test(std::memory_order_relaxed)) ;
//        }
//    }
//    void unlock() {
//        atomic_flag.clear(std::memory_order_release);
//    }
//};

class Spinlock {
    private: std::atomic<bool> my_lock = {false};
    
    public: void lock() {
        for (;;){
            if (!my_lock.exchange(true, std::memory_order_acquire)){
                break;
            }
            while (my_lock.load(std::memory_order_relaxed));
        }
    }
    
    void unlock() {my_lock.store(false);}
};

Spinlock mtx[5000];

// Function computes the pixel adjacency value to determine
// what type of bitmap is present for 2x2 or 2x2x2 patterns
float pixel_adjacency(int pos1[], int pos2[], int n_dims){
    float adj, abs_dist;
    abs_dist = 0.0;
    
    for(int i = 0; i < n_dims; i++){
        abs_dist += float(abs(pos1[i] - pos2[i]));
    }
    
    adj = sqrt(abs_dist);
    return adj;
}

//
void hex_decoder(char hex_char, int dec_rep){
    int intermediate;
    if(int(hex_char) > 65){
        intermediate = int('a') + 10;
    }
    else{
        intermediate = int('0');
    }
    dec_rep = int(hex_char) - intermediate;
}

// Method to decode bytes from little endian encoding for
// reading correct bitmap locations
int decode_little_endian_BMP(char *input_hexstring, int str_len, bool signed_int){
    int value = 0;
    return value;
}

// Integer version of decoding values in BMP file for things
int decode_little_endian_BMP_int(int *input_int, int str_len, bool signed_int){
    int value = 0;
    
    // Assert str_len is even; should always have
    // 2 hex_str characters for each byte!
    if(str_len % 2 == 1){
        fputs("str_len should be even\n", stderr);
        exit(1);
    }
    
    for(int i = 0; i < (str_len / 2); i++){
        // Adjusting value
        value += int(pow(16, (2*i))) * input_int[i];
    }
    
    if(signed_int){
        value = value - 16 ^ (str_len / 2);
    }
    
    return value;
}

// Function for argsorting the values of data throughout the
// operations.
void argsort(int data[], int locs[], int n_dims){
    int len_data = 0;
    // Adjust length of locs by n_dims
    for(int i = 0; i < n_dims; i++){
        if (len_data < 1){
            len_data = 2;
        }
        else{
            len_data = len_data * 2;
        }
    }
    
    // Sort the data and keep track of location/swap location
    // Using bubble sort
    // loop to access each array element
    for (int step = 0; step < len_data - 1; step++) {
        int swapped = 0;
        
        // loop to compare array elements
        for (int k = 0; k < len_data - step - 1; k++) {
            
            // compare two adjacent elements
            if (data[k] > data[k + 1]) {
                
                // swapping elements if elements
                // are not in the intended order
                int temp = data[k];
                data[k] = data[k + 1];
                data[k + 1] = temp;
                
                // swapping location elements (argsort)
                int temp_loc = locs[k];
                locs[k] = locs[k + 1];
                locs[k + 1] = temp_loc;
                
                swapped = 1;
            }
        }
        // If already sorted, stop sorting :)
        if (swapped == 0){
            break;
        }
    }
}

// Function determines the contribution of a 2x2 bitquad
// if a sublevel set is assumed.
void local_contr_2D_sub(int data[], int bitmap[][10]){
    // Initialize indices vector
    int inds[4];
    float adj;
    for(int j = 0; j < 4; j++){
        inds[j] = j;
    }
    
    // Copying data for pass-by-reference errors
    int temp_data[4];
    std::copy(&data[0], &data[4], &temp_data[0]);  // Tested copy, appears to be working
    
    // sort the data in ascending order
    argsort(temp_data, inds, 2);
    
    // Initialize location order for adjacency computation
    int loc1[2], loc2[2];
    loc1[0] = inds[0] / 2; // Compute row of data point
    loc1[1] = inds[0] % 2; // Compute column of data point
    loc2[0] = inds[1] / 2; // Compute row of data point
    loc2[1] = inds[1] % 2; // Compute column of data point
    
    // Compute adjacency or pixel 1 and 2 to differntiate a
    // vertex-connected component from an edge-connected component
    adj = pixel_adjacency(loc1, loc2, 2);
    
    // Update overall bitmap.
    // Row of bitmap = threshold value for level set
    // Col of bitmap = type of event (q0_start, q0_end, q1_start, q1_end, ..., q4_end)
    // Update q0 entry and exit from the system
    // DON'T NEED q0; will cause lots of problem when
    // parallelizing with locking mechanism.
    // bitmap[0][0]++;
    // bitmap[temp_data[0]][1]++;
    // Uncomment and adjust indices if you want q0
    // Also, you can deduce q0 behavior from q1!
    // And sum(qi) = # vertices at a given filtration level
    
    // Update q1 entry and exit from the system
    bitmap[temp_data[0]][0]++;
    bitmap[temp_data[1]][1]++;
    
    // Update q2 or qd entry and exit from the system
    if (adj < 1.1){
        bitmap[temp_data[1]][2]++;
        bitmap[temp_data[2]][3]++;
    }
    else{
        bitmap[temp_data[1]][4]++;
        bitmap[temp_data[2]][5]++;
    }
    
    // Update q3 entry and exit from the system
    bitmap[temp_data[2]][6]++;
    bitmap[temp_data[3]][7]++;
    
    // Update q4 entry to the system
    bitmap[temp_data[3]][8]++;
}

// Function determines the contribution of a 2x2 bitquad
// if a sublevel set is assumed.
void local_contr_2D_sub_flat(int data[], int bitmap[]){
    // Initialize indices vector
    int inds[4];
    float adj;
    for(int j = 0; j < 4; j++){
        inds[j] = j;
    }
    
    // Copying data for pass-by-reference errors
    int temp_data[4];
    std::copy(&data[0], &data[4], &temp_data[0]);  // Tested copy, appears to be working
    
    // sort the data in ascending order
    argsort(temp_data, inds, 2);
    
    // Initialize location order for adjacency computation
    int loc1[2], loc2[2];
    loc1[0] = inds[0] / 2; // Compute row of data point
    loc1[1] = inds[0] % 2; // Compute column of data point
    loc2[0] = inds[1] / 2; // Compute row of data point
    loc2[1] = inds[1] % 2; // Compute column of data point
    
    // Compute adjacency or pixel 1 and 2 to differntiate a
    // vertex-connected component from an edge-connected component
    adj = pixel_adjacency(loc1, loc2, 2);
    
    // Update overall bitmap.
    // Row of bitmap = threshold value for level set
    // Col of bitmap = type of event (q0_start, q0_end, q1_start, q1_end, ..., q4_end)
    // Update q0 entry and exit from the system
    // DON'T NEED q0; will cause lots of problem when
    // parallelizing with locking mechanism.
    // bitmap[0][0]++;
    // bitmap[temp_data[0]][1]++;
    // Uncomment and adjust indices if you want q0
    // Also, you can deduce q0 behavior from q1!
    // And sum(qi) = # vertices at a given filtration level
    
    // All indices of flatten array are index * 10 + element
    // i.e., if the data is 8 and causes a q1 to come in, then
    //       we increment the loss of q0 (element 81) and spawn
    //       of q1 (element 82)
    
    // Update q1 entry and exit from the system
    bitmap[temp_data[0] * 10 + 0]++;
    bitmap[temp_data[1] * 10 + 1]++;
    
    // Update q2 or qd entry and exit from the system
    if (adj < 1.1){
        bitmap[temp_data[1] * 10 + 2]++;
        bitmap[temp_data[2] * 10 + 3]++;
    }
    else{
        bitmap[temp_data[1] * 10 + 4]++;
        bitmap[temp_data[2] * 10 + 5]++;
    }
    
    // Update q3 entry and exit from the system
    bitmap[temp_data[2] * 10 + 6]++;
    bitmap[temp_data[3] * 10 + 7]++;
    
    // Update q4 entry to the system
    bitmap[temp_data[3] * 10 + 8]++;
}


// Function determines the contribution of a 2x2 bitquad
// if a sublevel set is assumed.
void local_contr_2D_sub_flat_spl(int data[], int bitmap[]){
    // Initialize indices vector
    int inds[4];
    float adj;
    for(int j = 0; j < 4; j++){
        inds[j] = j;
    }
    
    // Copying data for pass-by-reference errors
    int temp_data[4];
    std::copy(&data[0], &data[4], &temp_data[0]);  // Tested copy, appears to be working
    
    // sort the data in ascending order
    argsort(temp_data, inds, 2);
    
    // Initialize location order for adjacency computation
    int loc1[2], loc2[2];
    loc1[0] = inds[0] / 2; // Compute row of data point
    loc1[1] = inds[0] % 2; // Compute column of data point
    loc2[0] = inds[1] / 2; // Compute row of data point
    loc2[1] = inds[1] % 2; // Compute column of data point
    
    // Compute adjacency or pixel 1 and 2 to differntiate a
    // vertex-connected component from an edge-connected component
    adj = pixel_adjacency(loc1, loc2, 2);
    
    // Update overall bitmap.
    // Row of bitmap = threshold value for level set
    // Col of bitmap = type of event (q0_start, q0_end, q1_start, q1_end, ..., q4_end)
    // Update q0 entry and exit from the system
    // DON'T NEED q0; will cause lots of problem when
    // parallelizing with locking mechanism.
    // bitmap[0][0]++;
    // bitmap[temp_data[0]][1]++;
    // Uncomment and adjust indices if you want q0
    // Also, you can deduce q0 behavior from q1!
    // And sum(qi) = # vertices at a given filtration level
    
    // All indices of flatten array are index * 10 + element
    // i.e., if the data is 8 and causes a q1 to come in, then
    //       we increment the loss of q0 (element 81) and spawn
    //       of q1 (element 82)
    
    // Update q1 entry and exit from the system
    mtx[(temp_data[0] * 10 + 0) % 5000].lock();
    bitmap[temp_data[0] * 10 + 0]++;
    mtx[(temp_data[0] * 10 + 0) % 5000].unlock();
    
    mtx[(temp_data[1] * 10 + 1) % 5000].lock();
    bitmap[temp_data[1] * 10 + 1]++;
    mtx[(temp_data[1] * 10 + 1) % 5000].unlock();
    
    // Update q2 or qd entry and exit from the system
    if (adj < 1.1){
        mtx[(temp_data[1] * 10 + 2) % 5000].lock();
        bitmap[temp_data[1] * 10 + 2]++;
        mtx[(temp_data[1] * 10 + 2) % 5000].unlock();
        
        mtx[(temp_data[2] * 10 + 3) % 5000].lock();
        bitmap[temp_data[2] * 10 + 3]++;
        mtx[(temp_data[2] * 10 + 3) % 5000].unlock();
    }
    else{
        mtx[(temp_data[1] * 10 + 4) % 5000].lock();
        bitmap[temp_data[1] * 10 + 4]++;
        mtx[(temp_data[1] * 10 + 4) % 5000].unlock();
        
        mtx[(temp_data[2] * 10 + 5) % 5000].lock();
        bitmap[temp_data[2] * 10 + 5]++;
        mtx[(temp_data[2] * 10 + 5) % 5000].unlock();
    }
    
    // Update q3 entry and exit from the system
    //while(!mtx[(temp_data[2] * 10 + 6) % 5000].try_lock());
    mtx[(temp_data[2] * 10 + 6) % 5000].lock();
    bitmap[temp_data[2] * 10 + 6]++;
    mtx[(temp_data[2] * 10 + 6) % 5000].unlock();
    
    mtx[(temp_data[3] * 10 + 7) % 5000].lock();
    bitmap[temp_data[3] * 10 + 7]++;
    mtx[(temp_data[3] * 10 + 7) % 5000].unlock();
    
    // Update q4 entry to the system
    mtx[(temp_data[3] * 10 + 8) % 5000].lock();
    bitmap[temp_data[3] * 10 + 8]++;
    mtx[(temp_data[3] * 10 + 8) % 5000].unlock();
}


// Function determines the contribution of a 2x2 bitquad
// if a sublevel set is assumed.
void local_contr_2D_sub_flat_mtx(int data[], int bitmap[]){
    // Initialize indices vector
    int inds[4];
    float adj;
    for(int j = 0; j < 4; j++){
        inds[j] = j;
    }
    
    // Copying data for pass-by-reference errors
    int temp_data[4];
    std::copy(&data[0], &data[4], &temp_data[0]);  // Tested copy, appears to be working
    
    // sort the data in ascending order
    argsort(temp_data, inds, 2);
    
    // Initialize location order for adjacency computation
    int loc1[2], loc2[2];
    loc1[0] = inds[0] / 2; // Compute row of data point
    loc1[1] = inds[0] % 2; // Compute column of data point
    loc2[0] = inds[1] / 2; // Compute row of data point
    loc2[1] = inds[1] % 2; // Compute column of data point
    
    // Compute adjacency or pixel 1 and 2 to differntiate a
    // vertex-connected component from an edge-connected component
    adj = pixel_adjacency(loc1, loc2, 2);
    
    // Update overall bitmap.
    // Row of bitmap = threshold value for level set
    // Col of bitmap = type of event (q0_start, q0_end, q1_start, q1_end, ..., q4_end)
    // Update q0 entry and exit from the system
    // DON'T NEED q0; will cause lots of problem when
    // parallelizing with locking mechanism.
    // bitmap[0][0]++;
    // bitmap[temp_data[0]][1]++;
    // Uncomment and adjust indices if you want q0
    // Also, you can deduce q0 behavior from q1!
    // And sum(qi) = # vertices at a given filtration level
    
    // All indices of flatten array are index * 10 + element
    // i.e., if the data is 8 and causes a q1 to come in, then
    //       we increment the loss of q0 (element 81) and spawn
    //       of q1 (element 82)
    
    // Update q1 entry and exit from the system
    mtx[(temp_data[0] * 10 + 0) % 5000].lock();
    bitmap[temp_data[0] * 10 + 0]++;
    mtx[(temp_data[0] * 10 + 0) % 5000].unlock();
    
    mtx[(temp_data[1] * 10 + 1) % 5000].lock();
    bitmap[temp_data[1] * 10 + 1]++;
    mtx[(temp_data[1] * 10 + 1) % 5000].unlock();
    
    // Update q2 or qd entry and exit from the system
    if (adj < 1.1){
        mtx[(temp_data[1] * 10 + 2) % 5000].lock();
        bitmap[temp_data[1] * 10 + 2]++;
        mtx[(temp_data[1] * 10 + 2) % 5000].unlock();
        
        mtx[(temp_data[2] * 10 + 3) % 5000].lock();
        bitmap[temp_data[2] * 10 + 3]++;
        mtx[(temp_data[2] * 10 + 3) % 5000].unlock();
    }
    else{
        mtx[(temp_data[1] * 10 + 4) % 5000].lock();
        bitmap[temp_data[1] * 10 + 4]++;
        mtx[(temp_data[1] * 10 + 4) % 5000].unlock();
        
        mtx[(temp_data[2] * 10 + 5) % 5000].lock();
        bitmap[temp_data[2] * 10 + 5]++;
        mtx[(temp_data[2] * 10 + 5) % 5000].unlock();
    }
    
    // Update q3 entry and exit from the system
    //while(!mtx[(temp_data[2] * 10 + 6) % 5000].try_lock());
    mtx[(temp_data[2] * 10 + 6) % 5000].lock();
    bitmap[temp_data[2] * 10 + 6]++;
    mtx[(temp_data[2] * 10 + 6) % 5000].unlock();
    
    mtx[(temp_data[3] * 10 + 7) % 5000].lock();
    bitmap[temp_data[3] * 10 + 7]++;
    mtx[(temp_data[3] * 10 + 7) % 5000].unlock();
    
    // Update q4 entry to the system
    mtx[(temp_data[3] * 10 + 8) % 5000].lock();
    bitmap[temp_data[3] * 10 + 8]++;
    mtx[(temp_data[3] * 10 + 8) % 5000].unlock();
}

// Helper function to copy data to adequate input form
void copy_points(int locs[8][3], int temp_loc[3], int ind){
    for(int i = 0; i < 3; i++){
        temp_loc[i] = locs[ind][i];
    }
}

// Function determines the contribution of a 2x2x2 bitmap
// if a sublevel set is assumed.
void local_contr_3D_sub(int data[], int bitmap[][42]){
    // Initialize indices vector
    int inds[8];
    // Need to find adjacency for Q2, Q3, Q4, Q5, Q6
    float adj2, adj3, adj4, adj3e, adj2e;
    for(int j = 0; j < 8; j++){
        inds[j] = j;
    }
    
    // Copying data to make sure no pass-by-reference errors
    int temp_data[8];
    std::copy(&data[0], &data[8], &temp_data[0]);  // Tested copy, appears to be working
    
    //for(int j = 0; j < 8; j++){
    //    std::cout << data[j] << "\n";
    //}
    
    //std::cout << "\n";
    
    // sort the data in ascending order
    argsort(temp_data, inds, 3);
    
    // Initialize location order for adjacency computation
    int locs[8][3];
    int temp_loc1[3], temp_loc2[3];
    for(int i = 0; i < 8; i++){
        locs[i][0] = (inds[i] % 4) / 2; // Compute row of data point
        locs[i][1] = (inds[i] % 4) % 2; // Compute column of data point
        locs[i][2] = inds[i] / 4; // Compute depth of the data point (z-component)
    }
    
    // Compute voxel "total" adjacency for classification of the types of
    // voxel arrangements for 1 through 8 pixels
    
    // 2-voxel adjacency = adj(0, 1)
    copy_points(locs, temp_loc1, 0);
    copy_points(locs, temp_loc2, 1);
    adj2 = pixel_adjacency(temp_loc1, temp_loc2, 3); // adj(0, 1)
    
    // 3-voxel adjacency =  adj(0, 1) + adj(0, 2) + adj(1, 2)
    copy_points(locs, temp_loc2, 2);
    adj3 = adj2 + pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(0, 2)
    
    copy_points(locs, temp_loc1, 1);
    adj3 += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(1, 2)
    
    // 4-voxel adjacency = adj3 + adj(3, 0), adj(3, 1), adj(3, 2)
    copy_points(locs, temp_loc2, 3);
    adj4 = adj3 + pixel_adjacency(temp_loc1, temp_loc2, 3);  // adding adj(3, 1)
    
    copy_points(locs, temp_loc1, 2);
    adj4 += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(3, 2)
    
    copy_points(locs, temp_loc1, 0);
    adj4 += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(3, 0)
    
    // 6-voxel adjacency is the same as 2-empty-voxel adjacency
    // adj(8, 7)
    copy_points(locs, temp_loc1, 7);
    copy_points(locs, temp_loc2, 8);
    adj2e = pixel_adjacency(temp_loc1, temp_loc2, 3);  // adj(8, 7)
    
    // 5-voxel adjacency is the same as 3-empty-voxel adjacency
    // adj(8, 7) + adj (8, 6) + adj (7, 6)
    copy_points(locs, temp_loc1, 6);
    adj3e = adj2e + pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(8, 6)
    
    copy_points(locs, temp_loc2, 7);
    adj3e += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(7, 6)
    
    
    // Update overall bitmap.
    // Row of bitmap = threshold value for level set
    // Col of bitmap = type of event (q0_start, q0_end, q1_start, q1_end, ..., q4_end)
    // Update Q0i behavior
    // Nixxing Q0i addition; no information added and potential
    // for deadlocking with parallelism.
    // bitmap[0][0]++; // Q01 start
    // bitmap[temp_data[0]][1]++; // Q01 end
    // Uncomment above lines and adjust code below
    // to include Q0i
    
    // Update Q1i behavior
    bitmap[temp_data[0]][0]++; // Q11 start
    bitmap[temp_data[1]][1]++; // Q11 end
    
    // Update Q2i behavior
    if (adj2 < 1.2){
        bitmap[temp_data[1]][2]++; // Q21 start
        bitmap[temp_data[2]][3]++; // Q21 end
    }
    else if(adj2 < 1.5){
        bitmap[temp_data[1]][4]++; // Q22 start
        bitmap[temp_data[2]][5]++; // Q22 end
    }
    else{
        bitmap[temp_data[1]][6]++; // Q23 start
        bitmap[temp_data[2]][7]++; // Q23 end
    }
    
    // Update Q3i behavior
    if (adj3 < 4.0){
        bitmap[temp_data[2]][8]++; // Q31 start
        bitmap[temp_data[3]][9]++; // Q31 end
    }
    else if (adj3 < 4.2){
        bitmap[temp_data[2]][10]++; // Q32 start
        bitmap[temp_data[3]][11]++; // Q32 end
    }
    else{
        bitmap[temp_data[2]][12]++; // Q33 start
        bitmap[temp_data[3]][13]++; // Q33 end
    }
    
    // Update Q4i behavior
    if (adj4 < 7.0){
        bitmap[temp_data[3]][14]++; // Q41 start
        bitmap[temp_data[4]][15]++; // Q41 end
    }
    else if (adj4 < 7.4){
        bitmap[temp_data[3]][16]++; // Q42 start
        bitmap[temp_data[4]][17]++; // Q42 end
    }
    else if (adj4 < 7.7){
        bitmap[temp_data[3]][18]++; // Q43 start
        bitmap[temp_data[4]][19]++; // Q43 end
    }
    else if (adj4 < 8.2){
        bitmap[temp_data[3]][20]++; // Q44 start
        bitmap[temp_data[4]][21]++; // Q44 end
    }
    else if (adj4 < 8.4){
        bitmap[temp_data[3]][22]++; // Q45 start
        bitmap[temp_data[4]][23]++; // Q45 end
    }
    else{
        bitmap[temp_data[3]][24]++; // Q46 start
        bitmap[temp_data[4]][25]++; // Q46 end
    }
    
    // Update Q5i behavior
    if (adj3e < 4.0){
        bitmap[temp_data[4]][26]++; // Q51 start
        bitmap[temp_data[5]][27]++; // Q51 end
    }
    else if (adj3e < 4.2){
        bitmap[temp_data[4]][28]++; // Q52 start
        bitmap[temp_data[5]][29]++; // Q52 end
    }
    else{
        bitmap[temp_data[4]][30]++; // Q53 start
        bitmap[temp_data[5]][31]++; // Q53 end
    }
    
    // Update Q6i behavior
    if (adj2e < 1.2){
        bitmap[temp_data[5]][32]++; // Q61 start
        bitmap[temp_data[6]][33]++; // Q61 end
    }
    else if(adj2e < 1.5){
        bitmap[temp_data[5]][34]++; // Q62 start
        bitmap[temp_data[6]][35]++; // Q62 end
    }
    else{
        bitmap[temp_data[5]][36]++; // Q63 start
        bitmap[temp_data[6]][37]++; // Q63 end
    }
    
    // Update Q7i behavior
    bitmap[temp_data[6]][38]++; // Q71 start
    bitmap[temp_data[7]][39]++; // Q71 end
    
    // Update Q8i behavior
    bitmap[temp_data[7]][40]++; // Q81 start
}

void local_contr_3D_sub_flat(int data[], int bitmap[]){
    // Initialize indices vector
    int inds[8];
    // Need to find adjacency for Q2, Q3, Q4, Q5, Q6
    float adj2, adj3, adj4, adj3e, adj2e;
    for(int j = 0; j < 8; j++){
        inds[j] = j;
    }
    
    // Copying data to make sure no pass-by-reference errors
    int temp_data[8];
    std::copy(&data[0], &data[8], &temp_data[0]);  // Tested copy, appears to be working
    
    //for(int j = 0; j < 8; j++){
    //    std::cout << data[j] << "\n";
    //}
    
    //std::cout << "\n";
    
    // sort the data in ascending order
    argsort(temp_data, inds, 3);
    
    //for(int j = 0; j < 8; j++){
    //    std::cout << temp_data[j] << ", " << inds[j] << "\n";
    //}
    
    //std::cout << "\n";
    
    // Initialize location order for adjacency computation
    int locs[8][3];
    int temp_loc1[3], temp_loc2[3];
    for(int i = 0; i < 8; i++){
        locs[i][0] = (inds[i] % 4) / 2; // Compute row of data point
        locs[i][1] = (inds[i] % 4) % 2; // Compute column of data point
        locs[i][2] = inds[i] / 4; // Compute depth of the data point (z-component)
    }
    
    // Compute voxel "total" adjacency for classification of the types of
    // voxel arrangements for 1 through 8 pixels
    
    // 2-voxel adjacency = adj(0, 1)
    copy_points(locs, temp_loc1, 0);
    copy_points(locs, temp_loc2, 1);
    adj2 = pixel_adjacency(temp_loc1, temp_loc2, 3); // adj(0, 1)
    
    // 3-voxel adjacency =  adj(0, 1) + adj(0, 2) + adj(1, 2)
    copy_points(locs, temp_loc2, 2);
    adj3 = adj2 + pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(0, 2)
    
    copy_points(locs, temp_loc1, 1);
    adj3 += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(1, 2)
    
    // 4-voxel adjacency = adj3 + adj(3, 0), adj(3, 1), adj(3, 2)
    copy_points(locs, temp_loc2, 3);
    adj4 = adj3 + pixel_adjacency(temp_loc1, temp_loc2, 3);  // adding adj(3, 1)
    
    copy_points(locs, temp_loc1, 2);
    adj4 += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(3, 2)
    
    copy_points(locs, temp_loc1, 0);
    adj4 += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(3, 0)
    
    // 6-voxel adjacency is the same as 2-empty-voxel adjacency
    // adj(8, 7)
    copy_points(locs, temp_loc1, 6);
    copy_points(locs, temp_loc2, 7);
    adj2e = pixel_adjacency(temp_loc1, temp_loc2, 3);  // adj(7, 6)
    
    // 5-voxel adjacency is the same as 3-empty-voxel adjacency
    // adj(7, 6) + adj (7, 5) + adj (6, 5)
    copy_points(locs, temp_loc1, 5);
    adj3e = adj2e + pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(7, 5)
    
    copy_points(locs, temp_loc2, 6);
    adj3e += pixel_adjacency(temp_loc1, temp_loc2, 3); // adding adj(6, 5)
    
    
    // Update overall bitmap.
    // Row of bitmap = threshold value for level set
    // Col of bitmap = type of event (q0_start, q0_end, q1_start, q1_end, ..., q4_end)
    // Update Q0i behavior
    // Nixxing Q0i addition; no information added and potential
    // for deadlocking with parallelism.
    // bitmap[0][0]++; // Q01 start
    // bitmap[temp_data[0]][1]++; // Q01 end
    // Uncomment above lines and adjust code below
    // to include Q0i
    
    // Update Q1i behavior
    bitmap[temp_data[0] * 42 + 0]++; // Q11 start
    bitmap[temp_data[1] * 42 + 1]++; // Q11 end
    
    // Update Q2i behavior
    if (adj2 < 1.2){
        bitmap[temp_data[1] * 42 + 2]++; // Q21 start
        bitmap[temp_data[2] * 42 + 3]++; // Q21 end
    }
    else if(adj2 < 1.5){
        bitmap[temp_data[1] * 42 + 4]++; // Q22 start
        bitmap[temp_data[2] * 42 + 5]++; // Q22 end
    }
    else{
        bitmap[temp_data[1] * 42 + 6]++; // Q23 start
        bitmap[temp_data[2] * 42 + 7]++; // Q23 end
    }
    
    // Update Q3i behavior
    if (adj3 < 4.0){
        bitmap[temp_data[2] * 42 + 8]++; // Q31 start
        bitmap[temp_data[3] * 42 + 9]++; // Q31 end
    }
    else if (adj3 < 4.2){
        bitmap[temp_data[2] * 42 + 10]++; // Q32 start
        bitmap[temp_data[3] * 42 + 11]++; // Q32 end
    }
    else{
        bitmap[temp_data[2] * 42 + 12]++; // Q33 start
        bitmap[temp_data[3] * 42 + 13]++; // Q33 end
    }
    
    // Update Q4i behavior
    if (adj4 < 7.0){
        bitmap[temp_data[3] * 42 + 14]++; // Q41 start
        bitmap[temp_data[4] * 42 + 15]++; // Q41 end
    }
    else if (adj4 < 7.4){
        bitmap[temp_data[3] * 42 + 16]++; // Q42 start
        bitmap[temp_data[4] * 42 + 17]++; // Q42 end
    }
    else if (adj4 < 7.7){
        bitmap[temp_data[3] * 42 + 18]++; // Q43 start
        bitmap[temp_data[4] * 42 + 19]++; // Q43 end
    }
    else if (adj4 < 8.2){
        bitmap[temp_data[3] * 42 + 20]++; // Q44 start
        bitmap[temp_data[4] * 42 + 21]++; // Q44 end
    }
    else if (adj4 < 8.4){
        bitmap[temp_data[3] * 42 + 22]++; // Q45 start
        bitmap[temp_data[4] * 42 + 23]++; // Q45 end
    }
    else{
        bitmap[temp_data[3] * 42 + 24]++; // Q46 start
        bitmap[temp_data[4] * 42 + 25]++; // Q46 end
    }
    
    // Update Q5i behavior
    if (adj3e < 4.0){
        bitmap[temp_data[4] * 42 + 26]++; // Q51 start
        bitmap[temp_data[5] * 42 + 27]++; // Q51 end
    }
    else if (adj3e < 4.2){
        bitmap[temp_data[4] * 42 + 28]++; // Q52 start
        bitmap[temp_data[5] * 42 + 29]++; // Q52 end
    }
    else{
        bitmap[temp_data[4] * 42 + 30]++; // Q53 start
        bitmap[temp_data[5] * 42 + 31]++; // Q53 end
    }
    
    // Update Q6i behavior
    if (adj2e < 1.2){
        bitmap[temp_data[5] * 42 + 32]++; // Q61 start
        bitmap[temp_data[6] * 42 + 33]++; // Q61 end
    }
    else if(adj2e < 1.5){
        bitmap[temp_data[5] * 42 + 34]++; // Q62 start
        bitmap[temp_data[6] * 42 + 35]++; // Q62 end
    }
    else{
        bitmap[temp_data[5] * 42 + 36]++; // Q63 start
        bitmap[temp_data[6] * 42 + 37]++; // Q63 end
    }
    
    // Update Q7i behavior
    bitmap[temp_data[6] * 42 + 38]++; // Q71 start
    bitmap[temp_data[7] * 42 + 39]++; // Q71 end
    
    // Update Q8i behavior
    bitmap[temp_data[7] * 42 + 40]++; // Q81 start
}

// Function that takes an input image (2D) and performs the
// tasks required to determine a delta EC curve that can be
// used to compute the EC curve for TDA.
//
// Here, 'size_data' is the dilated size. i.e., if you are looking at a 3x3 grid, you
// will look at the 4x4 grid that surrounds the corners of the faces.
// input 'data' is the actual pixel values in a chunk that are required to evaluate the EC contributions
// Input 'max_val'
extern "C" void get_unit_contributions_2D(int *data, int contr_map[][10], int dim_x, int dim_y, int max_val, int size_data, int data_start_index, bool sup_level=false){
    // Iterate through data
    int row_d, col_d;  // Dilation true row and column
    int base_data_pos; // Data position to gather relative data from
    int temp_data[4];
    int dims_im[2];
    
    dims_im[0] = dim_x;
    dims_im[1] = dim_y;
    
    // std::cout << "Dims_im: " << dims_im[0] << ", " << dims_im[1] << "\n";
    
    // Printing data for debugging purposes
    //for(int i = 0; i < dim_x; i++){
    //    for(int j = 0; j < dim_y; j++){
    //        std::cout << data[i * dim_x + j] << " ";
    //    }
    //    std::cout << "\n";
    //}
    
    for(int i = data_start_index; i < size_data + data_start_index; i++){
        row_d = i / (dims_im[1] + 1);
        col_d = i % (dims_im[1] + 1);
        
        // Get base data position from the dilated row/column values
        base_data_pos = row_d * dims_im[1] + col_d;
        
        //std::cout << "Dilated position: " << i << "\n";
        //std::cout << "Base data position: " << base_data_pos << "\n";
        
        for(int j = 0; j < 4; j++){
            temp_data[j] = max_val + 1;
        }
        
        // Check for corner and edge pixels; treat the edge as large
        if(row_d != 0){
            if(col_d != 0){
                temp_data[0] = data[base_data_pos - dims_im[1] - 1];
            }
            if(col_d < dims_im[1]){
                temp_data[1] = data[base_data_pos - dims_im[1]];
            }
        }
        
        if(row_d < dims_im[0]){
            if(col_d != 0){
                temp_data[2] = data[base_data_pos - 1];
            }
            if(col_d < dims_im[1]){
                temp_data[3] = data[base_data_pos];
            }
        }
        
        //std::cout << "Size Data: " << size_data << "\n";
        //std::cout << "Start Index: " << data_start_index << "\n";
        //std::cout << "dims_im[0]: " << dims_im[0] << "\n";
        //std::cout << "dims_im[1]: " << dims_im[1] << "\n";
        //std::cout << "Data: (" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << ")\n";
        //std::cout << "Row_d: " << row_d << "\n";
        //std::cout << "Col_d: " << col_d << "\n";
        //std::cout << "0th ind: " << base_data_pos - dims_im[1] - 1 << "\n";
        //std::cout << "1st ind: " << base_data_pos - dims_im[1] << "\n";
        //std::cout << "2nd ind: " << base_data_pos - 1 << "\n";
        //std::cout << "3rd ind: " << base_data_pos << "\n";
        
        //for(int j = 0; j < 4; j++){
        //    std::cout << temp_data[j] << "\n";
        //}
        
        //std::cout << "\n";
        
        // Update EC contr_map with contributions from 'corners'
        local_contr_2D_sub(temp_data, contr_map);
    }
}

extern "C" void get_unit_contr_2D_flattened(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int size_data, int data_start_index, bool sup_level=false){
    // Iterate through data
    int row_d, col_d;  // Dilation true row and column
    int base_data_pos; // Data position to gather relative data from
    int temp_data[4];
    int dims_im[2];
    
    dims_im[0] = dim_x;
    dims_im[1] = dim_y;
    
    // std::cout << "Dims_im: " << dims_im[0] << ", " << dims_im[1] << "\n";
    
    // Printing data for debugging purposes
    //for(int i = 0; i < dim_x; i++){
    //    for(int j = 0; j < dim_y; j++){
    //        std::cout << data[i * dim_x + j] << " ";
    //    }
    //    std::cout << "\n";
    //}
    
    for(int i = data_start_index; i < size_data + data_start_index; i++){
        row_d = i / (dims_im[1] + 1);
        col_d = i % (dims_im[1] + 1);
        
        // Get base data position from the dilated row/column values
        base_data_pos = row_d * dims_im[1] + col_d;
        
        //std::cout << "Dilated position: " << i << "\n";
        //std::cout << "Base data position: " << base_data_pos << "\n";
        
        for(int j = 0; j < 4; j++){
            temp_data[j] = max_val + 1;
        }
        
        // Check for corner and edge pixels; treat the edge as large
        if(row_d != 0){
            if(col_d != 0){
                temp_data[0] = data[base_data_pos - dims_im[1] - 1];
            }
            if(col_d < dims_im[1]){
                temp_data[1] = data[base_data_pos - dims_im[1]];
            }
        }
        
        if(row_d < dims_im[0]){
            if(col_d != 0){
                temp_data[2] = data[base_data_pos - 1];
            }
            if(col_d < dims_im[1]){
                temp_data[3] = data[base_data_pos];
            }
        }
        
        //std::cout << "Size Data: " << size_data << "\n";
        //std::cout << "Start Index: " << data_start_index << "\n";
        //std::cout << "dims_im[0]: " << dims_im[0] << "\n";
        //std::cout << "dims_im[1]: " << dims_im[1] << "\n";
        //std::cout << "Data: (" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << ")\n";
        //std::cout << "Row_d: " << row_d << "\n";
        //std::cout << "Col_d: " << col_d << "\n";
        //std::cout << "0th ind: " << base_data_pos - dims_im[1] - 1 << "\n";
        //std::cout << "1st ind: " << base_data_pos - dims_im[1] << "\n";
        //std::cout << "2nd ind: " << base_data_pos - 1 << "\n";
        //std::cout << "3rd ind: " << base_data_pos << "\n";
        
        //for(int j = 0; j < 4; j++){
        //    std::cout << temp_data[j] << "\n";
        //}
        
        //std::cout << "\n";
        
        // Update EC contr_map with contributions from 'corners'
        local_contr_2D_sub_flat(temp_data, contr_map);
    }
}

// Function to compute EC contributions in a parallel manner.
// Will distribute computations as evenly as possible to
// the number of threads specified.
void compute_contr_parallel_CPU(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int data_start_index, int num_threads){
    
    int total_jobs = (dim_x + 1) * (dim_y + 1);
    int avg_jobs, last_jobs;
    std::thread* myThreads = new std::thread[num_threads];
    
    // Break up jobs
    avg_jobs = total_jobs / num_threads;
    last_jobs = total_jobs - (num_threads - 1) * avg_jobs;
    
    // Specify the threads
    for(int i = 0; i < num_threads; i++){
        if(i == (num_threads - 1)){
            myThreads[i] = std::thread(get_unit_contr_2D_flattened, data, contr_map, dim_x, dim_y, max_val, last_jobs, i * avg_jobs, false);
        }
        else{
            myThreads[i] = std::thread(get_unit_contr_2D_flattened, data, contr_map, dim_x, dim_y, max_val, avg_jobs, i * avg_jobs, false);
        }
    }
    
    // Join and clean up the threads!
    for(int i = 0; i < num_threads; i++){
        myThreads[i].join();
    }
    
    delete[] myThreads;
}

// Function to compute EC contributions in a parallel manner. But now add at the end
// with separate data entries for each thread.
// Will distribute computations as evenly as possible to
// the number of threads specified.
extern "C" void compute_contr_parallel_CPU_comb(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int data_start_index, int num_threads){
    
    int total_jobs = (dim_x + 1) * (dim_y + 1);
    int avg_jobs, last_jobs;
    std::thread* myThreads = new std::thread[num_threads];
    
    int** contr_maps = new int*[num_threads];
    
    for(int i = 0; i < num_threads; i++){
        contr_maps[i] = new int[(max_val + 2) * 10] ();
        for(int j = 0; j < (max_val + 2)*10; j++){
            contr_maps[i][j] = 0;
        }
    }
    
    // Break up jobs
    avg_jobs = total_jobs / num_threads;
    last_jobs = total_jobs - (num_threads - 1) * avg_jobs;
    
    // Specify the threads
    for(int i = 0; i < num_threads; i++){
        if(i == (num_threads - 1)){
            myThreads[i] = std::thread(get_unit_contr_2D_flattened, data, contr_maps[i], dim_x, dim_y, max_val, last_jobs, i * avg_jobs, false);
        }
        else{
            myThreads[i] = std::thread(get_unit_contr_2D_flattened, data, contr_maps[i], dim_x, dim_y, max_val, avg_jobs, i * avg_jobs, false);
        }
    }
    
    // Join and clean up the threads!
    for(int i = 0; i < num_threads; i++){
        myThreads[i].join();
    }
    
    
    for(int i = 0; i < (max_val + 2) * 10; i++){
        for(int j = 0; j < num_threads; j++){
            contr_map[i] += contr_maps[j][i];
        }
    }
    
    // Clean up arrays
    for(int i = 0; i < num_threads; i++){
        delete[] contr_maps[i];
    }
    
    // Clean up pointers
    delete[] contr_maps;
    delete[] myThreads;
}

void compute_contr_parallel_GPU(int *data, int *contr_map, int dim_x, int dim_y, int max_val, int data_start_index){
    
}

extern "C" void get_unit_contr_3D_flattened(int *data, int *contr_map, int dim_x, int dim_y, int dim_z, int max_val, int size_data, int data_start_index, bool sup_level=false){
    // Iterate through data
    int row_d, col_d, depth_d;  // Dilation true row and column
    int base_data_pos; // Data position to gather relative data from
    int temp_data[8];
    int dims_im[3];
    
    dims_im[0] = dim_x;
    dims_im[1] = dim_y;
    dims_im[2] = dim_z;
    
    // std::cout << "Dims_im: " << dims_im[0] << ", " << dims_im[1] << "\n";
    
    // Printing data for debugging purposes
    //for(int i = 0; i < dim_x; i++){
    //    for(int j = 0; j < dim_y; j++){
    //        std::cout << data[i * dim_x + j] << " ";
    //    }
    //    std::cout << "\n";
    //}
    
    for(int i = data_start_index; i < size_data + data_start_index; i++){
        // row_d = i / (dims_im[1] + 1);
        // col_d = i % (dims_im[1] + 1);
        row_d = (i % ((dims_im[0] + 1) * (dims_im[1] + 1))) / (dims_im[1] + 1);
        col_d = (i % ((dims_im[0] + 1) * (dims_im[1] + 1))) % (dims_im[1] + 1);
        depth_d = i / ((dims_im[0] + 1) * (dims_im[1] + 1));
        
        // Get base data position from the dilated row/column values
        base_data_pos = (row_d * dims_im[1] + col_d) + (depth_d * (dims_im[0] * dims_im[1]));
        
        //std::cout << "Dilated position: " << i << "\n";
        //std::cout << "Base data position: " << base_data_pos << "\n";
        
        for(int j = 0; j < 8; j++){
            temp_data[j] = max_val + 1;
        }
        
        // Check for corner and edge pixels; treat the edge as large
        // CHANGED THE POSITIONING OF DATA FROM 0, 4, 1, 5, 2, 6, 3, 7 to 0, 4, 2, 6, 1, 5, 3, 7
        if(row_d != 0){
            if(col_d != 0){
                if(depth_d != 0){
                    temp_data[0] = data[base_data_pos - dims_im[0] * dims_im[1] - dims_im[1] - 1];
                }
                if(depth_d < dims_im[2]){
                    temp_data[4] = data[base_data_pos - dims_im[1] - 1];
                }
            }
            if(col_d < dims_im[1]){
                if(depth_d != 0){
                    temp_data[1] = data[base_data_pos - dims_im[0] * dims_im[1] - dims_im[1]];
                }
                if(depth_d < dims_im[2]){
                    temp_data[5] = data[base_data_pos - dims_im[1]];
                }
            }
        }
        
        if(row_d < dims_im[0]){
            if(col_d != 0){
                if(depth_d != 0){
                    temp_data[2] = data[base_data_pos - dims_im[0] * dims_im[1] - 1];
                }
                if(depth_d < dims_im[2]){
                    temp_data[6] = data[base_data_pos - 1];
                }
            }
            if(col_d < dims_im[1]){
                if(depth_d != 0){
                    temp_data[3] = data[base_data_pos - dims_im[0] * dims_im[1]];
                }
                if(depth_d < dims_im[2]){
                    temp_data[7] = data[base_data_pos];
                }
            }
        }
        
        //std::cout << "Size Data: " << size_data << "\n";
        //std::cout << "Start Index: " << data_start_index << "\n";
        //std::cout << "dims_im[0]: " << dims_im[0] << "\n";
        //std::cout << "dims_im[1]: " << dims_im[1] << "\n";
        //std::cout << "dims_im[2]: " << dims_im[2] << "\n";
        //std::cout << "Data: (" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << ", " << data[4] << ", " << data[5] << ", " << data[6] << ", " << data[7] << ")\n";
        //std::cout << "i: " << i << "\n";
        //std::cout << "Row_d: " << row_d << "\n";
        //std::cout << "Col_d: " << col_d << "\n";
        //std::cout << "Depth_d: " << depth_d << "\n";
        //std::cout << "0th ind: " << base_data_pos - dims_im[0] * dims_im[1] - dims_im[1] - 1 << "\n";
        //std::cout << "1st ind: " << base_data_pos - dims_im[0] * dims_im[1] - dims_im[1] << "\n";
        //std::cout << "2nd ind: " << base_data_pos - dims_im[0] * dims_im[1] - 1 << "\n";
        //std::cout << "3rd ind: " << base_data_pos - dims_im[0] * dims_im[1] << "\n";
        //std::cout << "4th ind: " << base_data_pos - dims_im[1] - 1 << "\n";
        //std::cout << "5th ind: " << base_data_pos - dims_im[1] << "\n";
        //std::cout << "6th ind: " << base_data_pos - 1 << "\n";
        //std::cout << "7th ind: " << base_data_pos << "\n";
        
        // for(int j = 0; j < 8; j++){
        //    std::cout << temp_data[j] << " ";
        //}
        
        //std::cout << "\n";
        
        // Update EC contr_map with contributions from 'corners'
        local_contr_3D_sub_flat(temp_data, contr_map);
    }
}

extern "C" void compute_contr_parallel_CPU_comb_3D(int *data, int *contr_map, int dim_x, int dim_y, int dim_z, int max_val, int data_start_index, int num_threads){
    
    int total_jobs = (dim_x + 1) * (dim_y + 1) * (dim_z + 1);
    int avg_jobs, last_jobs;
    std::thread* myThreads = new std::thread[num_threads];
    
    int** contr_maps = new int*[num_threads];
    
    
    for(int i = 0; i < num_threads; i++){
        contr_maps[i] = new int[(max_val + 2) * 42];
        for(int j = 0; j < (max_val + 2)*42; j++){
            contr_maps[i][j] = 0;
        }
    }
    
    
    // Break up jobs
    avg_jobs = total_jobs / num_threads;
    last_jobs = total_jobs - (num_threads - 1) * avg_jobs;
    
    
    // Specify the threads
    for(int i = 0; i < num_threads; i++){
        if(i == (num_threads - 1)){
            myThreads[i] = std::thread(get_unit_contr_3D_flattened, data, contr_maps[i], dim_x, dim_y, dim_z, max_val, last_jobs, i * avg_jobs, false);
        }
        else{
            myThreads[i] = std::thread(get_unit_contr_3D_flattened, data, contr_maps[i], dim_x, dim_y, dim_z, max_val, avg_jobs, i * avg_jobs, false);
        }
    }
    
    // Join and clean up the threads!
    for(int i = 0; i < num_threads; i++){
        myThreads[i].join();
    }
    
    
    for(int i = 0; i < (max_val + 2) * 42; i++){
        for(int j = 0; j < num_threads; j++){
            contr_map[i] += contr_maps[j][i];
        }
    }
    
    // Clean up arrays
    for(int i = 0; i < num_threads; i++){
        delete[] contr_maps[i];
    }
    
    // Clean up pointers
    delete[] contr_maps;
    delete[] myThreads;
}


// Computing contributions with low memory by only reading in the relevant data
// at each vertex.
extern "C" void compute_contr_2D_low_mem(char *filename, int *contr_map, int max_val, long int start_val, long int total_jobs){
    // Read in the file
    FILE *image_file;
    char fileType[3];  // Type of file, checking for BM*
    int width_arr[4] = {0};  // Array to hold width byte values
    int height_arr[4] = {0};  // Array to hold the height byte values
    int byte_offset_arr[4] = {0};  // Array to hold the byte offset values
    int bits_per_pixel_arr[2] = {0};  // Array to hold the number of bits per pixel value (24 for RGB, 32 for ARGB)
    int temp_data[4];  // Array to hold data for the computation of EC contributions
    int width, height, byte_offset, bits_per_pixel, packing, row_length;  // Attribute integers
    int col_d, row_d;  // Integers to track positioning in the file
    long int curr_offset;
    bool no_seek = false;  // if we don't need to seek for the next one, we won't!
    
    int temp_data_holder[4] = {0};  // Temporary storage for data being read
    int temp_data_grabber = 0;  // Temporary int for data
    
    // Open the file and make sure it's valid
    image_file = fopen(filename, "rb");
    if(image_file == NULL){
        fputs("File error\n", stderr);
        exit(1);
    }
    
    // Make sure the filetype matches
    //fileType = (char*) malloc(sizeof(char)*2);
    fread(&fileType, 3, 1, image_file);
    
    if(fileType[0] != 'B' || fileType[1] != 'M'){
        fputs("File type error, should be BMP\n", stderr);
        exit(1);
    }
    
    // Extract values we need from the file
    // OFFSET NUMBER IN BYTES (bytes 10 - 13)
    fseek(image_file, 10, SEEK_SET);
    fread(&byte_offset_arr, 4, 1, image_file);
    // std::cout << "Byte offset array: \n";
    //for(int i = 0; i < 4; i++){
    //    std::cout << i << ": " << byte_offset_arr[i] << "\n";
    //}
    byte_offset = decode_little_endian_BMP_int(byte_offset_arr, 8);
    // WIDTH (bytes 18 - 21)
    fseek(image_file, 18, SEEK_SET);
    fread(&width_arr, 4, 1, image_file);
    width = decode_little_endian_BMP_int(width_arr, 8);
    // LENGTH (bytes 22 - 25)
    fread(&height_arr, 4, 1, image_file);
    height = decode_little_endian_BMP_int(height_arr, 8);
    // BITS PER PIXEL (bytes 28 - 29)
    fseek(image_file, 28, SEEK_SET);
    fread(&bits_per_pixel_arr, 2, 1, image_file);
    bits_per_pixel = decode_little_endian_BMP_int(bits_per_pixel_arr, 4);
    
    if(total_jobs == 0){
        total_jobs = (long int)(width + 1) * (long int)(height + 1);
    }
    // std::cout << "Total jobs: " << total_jobs << "\n";
    
    packing = (8 * width - (bits_per_pixel / 8) * width) % 4;
    
    // std::cout << "packing: " << packing << "\n";
    
    // std::cout << byte_offset << ", " << width << ", " << height << ", " << bits_per_pixel << "\n";
    
    // std::cout << filename << "\n";
    
    //for(int i = 0; i < width*height; i++){
    //    curr_col = i % width;
    //   curr_row = i / width;
        
    //    curr_offset = int((height - 1 - curr_row) * ((width * bits_per_pixel / 8) + packing) + int((bits_per_pixel / 8) * curr_col)) + int(byte_offset);
    //    fseek(image_file, curr_offset, SEEK_SET);
    //    for(int j = 0; j < int(bits_per_pixel / 3); j++){
    //        fread(&temp_data[j], 1, 1, image_file);
    //    }
        
    //    std::cout << "Col: " << curr_col << ", Row: " << curr_row << ", Offset: " << curr_offset << "\n";
    //    std::cout << "i value: " << i << ", data value read from file: (" << temp_data[0] << ", " << temp_data[1] << ", " << temp_data[2] << ")" << "\n";
    //}
    
    //std::cout << "Bits per pixel: " << (bits_per_pixel / 8) << "\n";
    //std::cout << "Row Length: " << ((width * bits_per_pixel / 8) + packing) << "\n";
    
    int RGB_offset = 0;
    // fpos_t pos;
    
    RGB_offset = (bits_per_pixel / 8 ) - 3;
    
    // for(long int i = 0; i < (long int)(width + 1) * (long int)(height + 1); i++){
    for(long int i = start_val; i < start_val + total_jobs; i++){
        row_d = int(i / (long int)(width + 1));
        col_d = int(i % (long int)(width + 1));
        
        row_length = ((width * bits_per_pixel / 8) + packing);
        
        // Get base data position from the dilated row/column values
        curr_offset = (height - row_d) * row_length + col_d * bits_per_pixel / 8 + byte_offset;
        
        //if(i % 1000000000 == 0){
        //    std::cout << "Total of " << i << " elements complete.\n";
        //}
        
        //std::cout << "Position starting top right: " << (curr_offset) << "\n";
        //std::cout << "Col, Row: (" << col_d << ", " << row_d << ")\n";
        
        //std::cout << "Dilated position: " << i << "\n";
        
        for(int j = 0; j < 4; j++){
            temp_data[j] = max_val + 1;
        }
        
        // Check for corner and edge pixels; treat the edge as large
        if(row_d != 0){
            if(col_d != 0){
                fseek(image_file, (curr_offset - (bits_per_pixel / 8)), SEEK_SET);
                no_seek = true;
                for(int j = 0; j < (bits_per_pixel / 8); j++){
                    fread(&temp_data_grabber, 1, 1, image_file);
                    // std::cout << temp_data_grabber << " ";
                    temp_data_holder[j] = temp_data_grabber;
                }
                // std::cout << "\n";
                // 0.299R + 0.587G + 0.114B
                // std::cout << "Math 0: " << int(ceil(0.299 * double(temp_data_holder[0 + RGB_offset]) + 0.587 * double(temp_data_holder[1 + RGB_offset]) + 0.114 * double(temp_data_holder[2 + RGB_offset]))) << "\n";
                temp_data[0] = int(ceil(0.299 * temp_data_holder[2] + 0.587 * temp_data_holder[1] + 0.114 * temp_data_holder[0]));;
            }
            if(col_d < width){
                if(!no_seek){
                    fseek(image_file, curr_offset, SEEK_SET);
                }
                for(int j = 0; j < (bits_per_pixel / 8); j++){
                    fread(&temp_data_grabber, 1, 1, image_file);
                    temp_data_holder[j] = temp_data_grabber;
                }
                temp_data[1] = int(ceil(0.299 * temp_data_holder[2] + 0.587 * temp_data_holder[1] + 0.114 * temp_data_holder[0]));;
            }
        }
        
        no_seek = false;
        
        if(row_d < height){
            if(col_d != 0){
                fseek(image_file, (curr_offset - row_length - (bits_per_pixel / 8)), SEEK_SET);
                no_seek = true;
                for(int j = 0; j < (bits_per_pixel / 8); j++){
                    fread(&temp_data_grabber, 1, 1, image_file);
                    temp_data_holder[j] = temp_data_grabber;
                }
                temp_data[2] = int(ceil(0.299 * temp_data_holder[2] + 0.587 * temp_data_holder[1] + 0.114 * temp_data_holder[0]));;
            }
            if(col_d < width){
                if(!no_seek){
                    fseek(image_file, (curr_offset - row_length), SEEK_SET);
                }
                for(int j = 0; j < (bits_per_pixel / 8); j++){
                    fread(&temp_data_grabber, 1, 1, image_file);
                    temp_data_holder[j] = temp_data_grabber;
                }
                temp_data[3] = int(ceil(0.299 * temp_data_holder[2] + 0.587 * temp_data_holder[1] + 0.114 * temp_data_holder[0]));;
            }
        }
        no_seek = false;
        
        local_contr_2D_sub_flat(temp_data, contr_map);
    }
    
    fclose(image_file);
    // delete[] temp_data_holder;
}

// Parallel version of the previous function
extern "C" void compute_contr_2D_low_mem_parallel(char *filename, int *contr_map, int max_val, int num_threads){
    // Read in the file
    FILE *image_file;
    char fileType[3];  // Type of file, checking for BM*
    int width_arr[4] = {0};  // Array to hold width byte values
    int height_arr[4] = {0};  // Array to hold the height byte values
    int byte_offset_arr[4] = {0};  // Array to hold the byte offset values
    int bits_per_pixel_arr[2] = {0};  // Array to hold the number of bits per pixel value (24 for RGB, 32 for ARGB)
    int width, height, byte_offset, bits_per_pixel;  // Attribute integers
    long int total_jobs, avg_jobs, last_jobs;
    
    // Parallel things
    std::thread* myThreads = new std::thread[num_threads];
    
    int** contr_maps = new int*[num_threads];
    
    for(int i = 0; i < num_threads; i++){
        contr_maps[i] = new int[(max_val + 2) * 42] ();
        for(int j = 0; j < (max_val + 2); j++){
            contr_maps[i][j] = 0;
        }
    }
    
    // Open the file and make sure it's valid
    image_file = fopen(filename, "rb");
    if(image_file == NULL){
        fputs("File error\n", stderr);
        exit(1);
    }
    
    // Make sure the filetype matches
    //fileType = (char*) malloc(sizeof(char)*2);
    fread(&fileType, 3, 1, image_file);
    
    if(fileType[0] != 'B' || fileType[1] != 'M'){
        fputs("File type error, should be BMP\n", stderr);
        exit(1);
    }
    
    // Extract values we need from the file
    // OFFSET NUMBER IN BYTES (bytes 10 - 13)
    fseek(image_file, 10, SEEK_SET);
    fread(&byte_offset_arr, 4, 1, image_file);
    //std::cout << "Byte offset array: \n";
    //for(int i = 0; i < 4; i++){
    //    std::cout << i << ": " << byte_offset_arr[i] << "\n";
    //}
    byte_offset = decode_little_endian_BMP_int(byte_offset_arr, 8);
    // WIDTH (bytes 18 - 21)
    fseek(image_file, 18, SEEK_SET);
    fread(&width_arr, 4, 1, image_file);
    width = decode_little_endian_BMP_int(width_arr, 8);
    // LENGTH (bytes 22 - 25)
    fread(&height_arr, 4, 1, image_file);
    height = decode_little_endian_BMP_int(height_arr, 8);
    // BITS PER PIXEL (bytes 28 - 29)
    fseek(image_file, 28, SEEK_SET);
    fread(&bits_per_pixel_arr, 2, 1, image_file);
    bits_per_pixel = decode_little_endian_BMP_int(bits_per_pixel_arr, 4);
    
    total_jobs = (long int)(width + 1) * (long int)(height + 1);
    
    // Break up jobs
    avg_jobs = total_jobs / num_threads;
    last_jobs = total_jobs - (num_threads - 1) * avg_jobs;
    
    // Specify the threads
    for(int i = 0; i < num_threads; i++){
        if(i == (num_threads - 1)){
            myThreads[i] = std::thread(compute_contr_2D_low_mem, filename, contr_maps[i], max_val, i * avg_jobs, last_jobs);
        }
        else{
            myThreads[i] = std::thread(compute_contr_2D_low_mem, filename, contr_maps[i], max_val, i * avg_jobs, avg_jobs);
        }
    }
    
    // Join and clean up the threads!
    for(int i = 0; i < num_threads; i++){
        myThreads[i].join();
    }
    
    
    for(int i = 0; i < (max_val + 2) * 42; i++){
        for(int j = 0; j < num_threads; j++){
            contr_map[i] += contr_maps[j][i];
        }
    }
    
    // Clean up arrays
    for(int i = 0; i < num_threads; i++){
        delete[] contr_maps[i];
    }
    
    // Clean up pointers
    delete[] contr_maps;
    delete[] myThreads;
}
