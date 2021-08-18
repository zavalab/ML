#!/bin/bash
cp /staging/sjiang87/plastic.tar.gz ./
tar -zxvf plastic.tar.gz
rm *.tar.gz
python3 train_2d_fast_multilabel.py --lr=$1 --bs=$2 --mcp=$3 --nf=$4 --im=$5
python3 train_2d_fast_revalidate.py --lr=$1 --bs=$2 --mcp=$3 --nf=$4 --im=$5

rm *.py
rm fast_with_pc.pickle
rm fast_with_pc_25.pickle
rm fast_with_pc_50.pickle
rm fast_with_pc_75.pickle
rm fast_with_pc_100.pickle
rm fast_with_pc_multilabel.pickle
rm revalidate_fast_with_pc_25_test.pickle
rm revalidate_fast_with_pc_25.pickle
rm revalidate_fast_with_pc_50_test.pickle
rm revalidate_fast_with_pc_50.pickle
rm revalidate_fast_with_pc_75_test.pickle
rm revalidate_fast_with_pc_75.pickle
rm revalidate_fast_with_pc_100_test.pickle
rm revalidate_fast_with_pc_100.pickle
rm revalidate_fast_with_pc_test.pickle
rm revalidate_fast_with_pc.pickle
rm docker_stderror
