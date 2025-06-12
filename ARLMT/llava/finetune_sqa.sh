#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

deepspeed train/train_mem.py \
    --deepspeed /home/guoyunfei/projects/LLaVA-main/scripts/zero3.json \
    --model_name_or_path /data0/GYF/offline_model/LLaVA-Med/pvqa-9epoch_delta/ \
    --version v1 \
    --data_path /data0/GYF/data/llava-med/PathVQA/split_data/train/llava-med-PathVQA-train2.json \
    --image_folder /data0/GYF/data/llava-med/PathVQA/split_data/train/images/ \
    --vision_tower /data0/GYF/offline_model/openai/clip-vit-large-patch14-336/ \
    --pretrain_mm_mlp_adapter /home/guoyunfei/projects/LLaVA-main/llava/checkpoints/llava-v1.6-vicuna-13b-med+blip558k-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/guoyunfei/projects/LLaVA-main/llava/checkpoints/pvqa-9epoch_delta-SLAKE_QCM_LEA-12e \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
