#!/bin/bash

list="lego"
# list="my_386"

# list="chair drums ficus hotdog lego materials mic ship"
# root_dir="datasets/nerf_synthetic/"
root_dir="/data/3D_data/nerf_synthetic/"
# root_dir="/data/3D_data/mocap_hj/"

# /data/3D_data/mocap_hj/my_377/transforms_train.json
exp_name="3dgs-neus-best-mask-HP10-no-reg-test_normal_map_results"
# exp_name="241009_hj_results"

for i in $list; do
if [ "$i" = "materials" ]; then
    HP=100
else
    HP=10
fi

python train_hj.py --eval \
-s ${root_dir}${i} \
-t bind \
-m output/NeRF_Syn/${i}/${exp_name} \
--HP 10 \
--N_tri 3 \
--iteration 30000 \
--lambda_mask_entropy 0.1 \
--lambda_normal_render_depth 0.01 \
--densification_interval 500 \
--save_training_vis
done

# chkpnt20000