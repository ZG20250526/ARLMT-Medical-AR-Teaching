torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    train/train_mem.py \
    --model_name_or_path /data0/GYF/offline_model/LLaVA-Med/output/vqa-rad-delta/ \
    --data_path /data0/GYF/data/llava-med/vqa-rad/split_data/train/triplets.json \
    --image_folder /data0/GYF/data/llava-med/vqa-rad/split_data/train/images \
    --vision_tower /data0/GYF/offline_model/openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir /home/guoyunfei/projects/llava-med-train/llava/results/vqa-rad-delta/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb    

    # --pretrain_mm_mlp_adapter /home/chunyl/research/models/llava/LLaVA-13b-pretrain-projector-v0/LLaVA-13b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
