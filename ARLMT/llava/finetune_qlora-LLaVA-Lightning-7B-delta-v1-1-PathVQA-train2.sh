#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

deepspeed /home/guoyunfei/projects/LLaVA-main/llava/train/train_mem.py \
    --deepspeed /home/guoyunfei/projects/LLaVA-main/scripts/zero2.json \
    --lora_enable True \
    --bits 4 \
    --model_name_or_path /data0/GYF/offline_model/liuhaotian/LLaVA-Lightning-7B-delta-v1-1 \
    --version v1 \
    --data_path /data0/GYF/data/llava-med/PathVQA/split_data/train/llava-med-PathVQA-train2.json \
    --image_folder /data0/GYF/data/llava-med/PathVQA/split_data/train/images/ \
    --vision_tower /data0/GYF/offline_model/openai/clip-vit-large-patch14-336/ \
    --pretrain_mm_mlp_adapter /home/guoyunfei/projects/LLaVA-main/llava/checkpoints/llava-v1.6-vicuna-13b-med+blip558k-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/guoyunfei/projects/LLaVA-main/llava/checkpoints/20250115-llama-7b-pvqa-9epoch_delta-SLAKE_QCM_LEA-12efinetune_qlora \
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
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
