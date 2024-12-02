# #!/bin/bash

# # list= "lego"
# # list="my_386        "
# list="ours"

# # list="chair drums ficus hotdog lego materials mic ship"
# # root_dir="datasets/nerf_synthetic/"
# # root_dir= "/data/3D_data/nerf_synthetic/"
# root_dir="/data/3D_data/mocap_hj/"

# # /data/3D_data/mocap_hj/my_377/transforms_train.json

# # exp_name="241122_3dgs-neus-best-mask-HP10-no-reg-test_normal_map_results"
# exp_name="241128_2_hj"

# # exp_name="mocap_normal_map_results"

# for i in $list; do
# if [ "$i" = "materials" ]; then
#     HP=100
# else
#     HP=10
# fi

# python train_hj_241118.py --eval \
# -s ${root_dir}${i} \
# -t bind \
# -m output/1016_mocap/${i}/${exp_name} \
# --HP 10 \
# --N_tri 3 \
# --iteration 20000 \
# --lambda_mask_entropy 0.1 \
# --lambda_normal_render_depth 0.01 \
# --densification_interval 500 \
# --save_training_vis
# done

# # chkpnt20000

#!/bin/bash

# CUDA device 설정
export CUDA_VISIBLE_DEVICES=1  # 사용할 GPU 번호 지정

# list= "lego"
# list="my_386"
list="ours"

# list="chair drums ficus hotdog lego materials mic ship"
# root_dir="datasets/nerf_synthetic/"
# root_dir= "/data/3D_data/nerf_synthetic/"
root_dir="/data/3D_data/mocap_hj/"

# /data/3D_data/mocap_hj/my_377/transforms_train.json

# exp_name="241122_3dgs-neus-best-mask-HP10-no-reg-test_normal_map_results"
exp_name="241202_hj"

# exp_name="mocap_normal_map_results"

for i in $list; do
if [ "$i" = "materials" ]; then
    HP=100
else
    HP=10
fi

python train_hj_241118.py --eval \
-s ${root_dir}${i} \
-t bind \
-m output/1016_mocap/${i}/${exp_name} \
--HP 10 \
--N_tri 3 \
--iteration 20000 \
--lambda_mask_entropy 0.1 \
--lambda_normal_render_depth 0.01 \
--lambda_depth 0.1 \
--lambda_normal_mvs_depth 0.01 \
--lambda_dssim 0.2 \
--densification_interval 500 \
--save_training_vis
done

# chkpnt20000