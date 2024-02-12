#!/bin/bash

run_0.25(){
    local zoom_ratio=$1
    for gamma in 2 3 4 5; do
        for lr in 0.001 0.0005 0.005; do
            for fold in $(seq 0 4); do
                CUDA_VISIBLE_DEVICES=0 python ./src/train_cnn.py --fold $fold --zoom_ratio $zoom_ratio --gamma $gamma --lr $lr
            done
        done
    done
}

run_nofocal(){
    local zoom_ratio=$1
    for gamma in 0; do
        for lr in 0.001; do
            for fold in $(seq 0 4); do
                CUDA_VISIBLE_DEVICES=0 python ./src/train_cnn.py --fold $fold --zoom_ratio $zoom_ratio --gamma $gamma --lr $lr
            done
        done
    done
}

run_other_res(){
    local zoom_ratio=$1
    for gamma in 2; do
        for lr in 0.001; do
            for fold in $(seq 0 4); do
                CUDA_VISIBLE_DEVICES=0 python ./src/train_cnn.py --fold $fold --zoom_ratio $zoom_ratio --gamma $gamma --lr $lr
            done
        done
    done
}

run_nofocal 0.25
run_0.25 0.25
run_other_res 0.125
run_other_res 0.5