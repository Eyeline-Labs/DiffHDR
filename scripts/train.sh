#!/usr/bin/env bash
# Training: LoRA on VACE (Wan-2.1 14B), hybrid with image, exposure masks, log encoding

export NCCL_TIMEOUT=1800

# Set these to your data root
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to your dataset base path}"

accelerate launch train_hdr.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$DATA_ROOT/metadata/metadata_synthetic_train.csv" \
  --prompt_path "$DATA_ROOT/metadata/prompt_train.csv" \
  --over_exposure_prompt_path "$DATA_ROOT/metadata/prompt_over_train.csv" \
  --under_exposure_prompt_path "$DATA_ROOT/metadata/prompt_under_train.csv" \
  --data_file_keys "video,vace_video,vace_video_mask,vace_reference_image" \
  --dataset_repeat 1000 \
  --height 720 \
  --width 1280 \
  --num_frames 33 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "models/DiffHDR" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "vace_video,vace_video_mask,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --is_exr \
  --log_every_n_steps 20 \
  --exp_name "DiffHDR" \
  --save_every_n_epochs 1 \
  --pin_memory \
  --dataset_num_workers 8 \
  --prefetch_factor 2 \
  --save_steps 400 \
  --max_train_steps 10000 \
  --use_exposured_gt \
  --srgb_to_lg \
  --add_exposure_mask \
  --use_under_expose_mask \
  --use_gradient_checkpointing \
  --add_noise_to_image \
  --add_noise_prob 0.05 \
  --use_exposure_augmentation \
  --use_vace_reference_image \
  --use_hybrid_training_with_image \
  --use_exposure_prompt
