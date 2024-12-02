#!/bin/bash

# list="lego"
list="my_387"

# list="chair drums ficus hotdog lego materials mic ship"
# root_dir="/data/3D_data/nerf_synthetic/"
root_dir="/data/3D_data/mocap_hj/my_387/colmap"

exp_name="241010_original_results"

for i in $list; do
python train.py \
-s ${root_dir}${i} \
-t render \
-m output/NeRF_Syn/${i}/3dgs \
-c output/NeRF_Syn/${i}/3dgs/chkpnt30000.pth \
--iteration 30000 \
--lambda_normal_render_depth 0.01 \
--lambda_mask_entropy 0.1 \
--densification_interval 500 \
--save_training_vis
done
