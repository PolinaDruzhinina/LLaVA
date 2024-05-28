#!/bin/bash
# --include localhost:5
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --compressed \
    --compressed_model_ckpt ./checkpoints/compressed_model_vq_vae \
    --compressed_model_visual_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
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
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb

#  --deepspeed /home/p.druzhinina/MMDCOT/LLaVA/scripts/zero2.json   --compressed_model_visual_adapter /home/p.druzhinina/MMDCOT/LLaVA/checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin --compressed --compressed_model_ckpt /home/p.druzhinina/MMDCOT/LLaVA/checkpoints/compressed_model_vq_vae  --model_name_or_path lmsys/vicuna-13b-v1.5 --version plain --data_path /home/p.druzhinina/MMDCOT/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json --image_folder /home/p.druzhinina/MMDCOT/LLaVA/playground/data/LLaVA-Pretrain/images --vision_tower openai/clip-vit-large-patch14-336 --mm_projector_type mlp2x_gelu --tune_mm_mlp_adapter True --mm_vision_select_layer -2 --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir /home/p.druzhinina/MMDCOT/LLaVA/checkpoints/llava-v1.5-13b-pretrain --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 24000 --save_total_limit 1 --learning_rate 1e-3 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 1 --lazy_preprocess True --report_to wandb