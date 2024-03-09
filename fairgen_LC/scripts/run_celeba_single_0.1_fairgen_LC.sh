#!/bin/bash

python3 ../src/BigGAN/train.py \
--shuffle --batch_size 16 --v_batch_size 8 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --num_D_v_steps 1 \
--G_lr 5e-5 --D_lr 2e-4 --D_v_lr 2e-4 \
--dataset CA64 \
--data_root ../data \
--base_root /media/usb_media/fairgen_ours \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 --D_v_attn 0 \
--G_init N02 --D_init N02 --D_v_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 1000 --test_every 1000 \
--num_best_copies 5 --num_save_copies 1 \
--loss_type hinge_dis_linear_gen \
--v_loss_type hinge_dis_linear_gen \
--num_epochs 150 --start_eval 10 \
--bias 90_10 --perc 0.1 \
--GPU_main 0 --nGPU 1 \
--lambda_ 0.9 \
--LC 0.05 --ema_losses_decay 0.9 --ema_losses_start 1000 \
--seed 777 \
--intra_FID --FID_off \
--name_suffix celeba_single_0.1_fairgen_LC