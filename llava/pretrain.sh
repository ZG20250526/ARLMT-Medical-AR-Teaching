#!/bin/bash
#TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \

deepspeed /home/guoyunfei/projects/LLaVA-main/llava/train/train_mem.py \
    --deepspeed /home/guoyunfei/projects/LLaVA-main/scripts/zero2.json \
    --model_name_or_path /data0/GYF/offline_model/liuhaotian/llava-v1.6-vicuna-13b \
    --version plain \
    --data_path /data0/GYF/data/llava/llava_med_alignment_10k+blip-2024090301.json  \
    --image_folder /data0/GYF/data/llava/LLaVA-Pretrain/images \
    --vision_tower /data0/GYF/offline_model/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/guoyunfei/projects/LLaVA-main/llava/checkpoints/llava-v1.6-vicuna-13b-med+blip558k-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
