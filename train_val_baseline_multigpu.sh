#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ./logs/%j_out.txt
#SBATCH -e ./logs/%j_err.txt
#SBATCH --gres=gpu:8

dataset=ntu_rgbd120  # ntu_rgbd60 / ntu_rgbd120 / nw_ucla / uwa3d
num_frames=16
img_size=224
modality=rgb_depth_ir
evaluation=cross_subject
model=baseline
model_size=small
fusion=sum  # sum / concat / transformer
modulation_ratio=1.0
end_epoch=200
lr=1e-4
bs=8
eval_bs=8
checkpoint=/save/neptune736.ckpt  
checkpoint_r=/save/neptune579.ckpt
checkpoint_d=/save/neptune582.ckpt
checkpoint_i=/save/neptune580.ckpt
checkpoint_s=/save/neptune583.ckpt

torchrun \
--master_port=$port \
--nproc_per_node=8 train_val_baseline_multigpu.py \
--dataset ${dataset} \
--num_frames ${num_frames} \
--img_size ${img_size} \
--modality ${modality} \
--evaluation ${evaluation} \
--model ${model} \
--model_size ${model_size} \
--fusion ${fusion} \
--end_epoch ${end_epoch} \
--lr ${lr} \
--bs ${bs} \
--eval_bs ${eval_bs} \
--checkpoint ${checkpoint} \
--checkpoint_r ${checkpoint_r} \
--checkpoint_d ${checkpoint_d} \
--checkpoint_i ${checkpoint_i} \
--checkpoint_s ${checkpoint_s} \
--temporal_augmentation \
--imagenet_pretrained \
--use_neptune
# --resume \
# --eval_untrained \
# --resume_all \
# --debug
# --use_gradient_modulation \