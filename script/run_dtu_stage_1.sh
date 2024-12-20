#!/bin/bash

root_dir="/data/3D_data/data_dtu/DTU_scan"
# list="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
list="118"

for i in $list
do
python train.py --eval \
-s ${root_dir}${i} \
-m output/DTU/${i}/3dgs \
--lambda_normal_render_depth 0.01 \
--lambda_mask_entropy 0.1 \
--lambda_depth 1 \
--iteration 10000 \
--lambda_normal_mvs_depth 0.01 \
--densification_interval 500 \
--save_training_vis

done