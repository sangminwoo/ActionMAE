#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ./logs/%j_out.txt
#SBATCH -e ./logs/%j_err.txt
#SBATCH --gres=gpu:8

dataset=ntu_rgbd60  # ntu_rgbd60 / ntu_rgbd120 / nw_ucla / uwa3d
num_frames=16
img_size=224
modality=rgb_depth
evaluation=cross_subject
model=actionmae
model_size=small
fusion=sum  # sum / concat / transformer
num_mem_token=4
set_loss_mask=1
set_loss_label=1
end_epoch=200
lr=1e-4
bs=8
eval_bs=8
checkpoint=/save/neptune656.ckpt  
checkpoint_r=/save/neptune579.ckpt
checkpoint_d=/save/neptune582.ckpt
checkpoint_i=/save/neptune580.ckpt
checkpoint_s=/save/neptune583.ckpt

torchrun \
--master_port=$port \
--nproc_per_node=8 train_val_actionmae_multigpu.py \
--dataset ${dataset} \
--num_frames ${num_frames} \
--img_size ${img_size} \
--modality ${modality} \
--evaluation ${evaluation} \
--model ${model} \
--model_size ${model_size} \
--fusion ${fusion} \
--num_mem_token ${num_mem_token} \
--set_loss_mask ${set_loss_mask} \
--set_loss_label ${set_loss_label} \
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
# --resume_all \
# --eval_untrained \
# --debug \
