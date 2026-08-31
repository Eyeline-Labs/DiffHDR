# Simplified training module: removes FiLM, hdr_ctrl, exposure routing, VAE training,
# channel weighting, highlight loss, x0 loss, chroma loss, inverse variance.

import torch, os, json
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.data.hdr_video import HDRVideoDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.lora_utils import resolve_vace_indices, make_lora_targets_by_keywords


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        srgb_to_lg=False,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for i in model_id_with_origin_paths:
                if "Wan2.1_VAE.pth" in i:
                    model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=torch.float32)]
                else:
                    model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1])]

        local_model_path = "models"
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            vae_dtype=torch.float32,
            device="cpu",
            model_configs=model_configs,
            local_model_path=local_model_path,
        )

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Add LoRA to the base models
        if lora_base_model is not None:
            chosen_indices = resolve_vace_indices(self.pipe.vace, "", "")
            if lora_base_model == "vace" and chosen_indices:
                target_modules = make_lora_targets_by_keywords(self.pipe.vace, chosen_indices,
                                                        keywords=lora_target_modules.split(","))
            else:
                target_modules = lora_target_modules.split(",")
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=target_modules,
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

        # Configs for HDR video training
        self.srgb_to_lg = srgb_to_lg

    def forward_preprocess(self, data, is_exr=False):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1] if not is_exr else data["video"][0].shape[1],
            "width": data["video"][0].size[0] if not is_exr else data["video"][0].shape[2],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "is_exr": is_exr,
            # For VACEVideo Transform
            "srgb_to_lg": self.srgb_to_lg,
            "binary_in_reactive_mask": False,
        }

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None, is_exr=False):
        if inputs is None: inputs = self.forward_preprocess(data, is_exr=is_exr)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = HDRVideoDataset(
        args=args,
        used_data_percentage=args.used_data_percentage,
        prompt_path=args.prompt_path,
        use_exposure_augmentation=args.use_exposure_augmentation,
        srgb_to_lg=args.srgb_to_lg,
        use_exposured_gt=args.use_exposured_gt,
        add_exposure_mask=args.add_exposure_mask,
        add_noise_prob=args.add_noise_prob,
        use_under_expose_mask=args.use_under_expose_mask,
        use_exposure_prompt=args.use_exposure_prompt,
        exposure_prompt_path=args.exposure_prompt_path,
        use_vace_reference_image=args.use_vace_reference_image,
        use_vace_reference_image_prob=args.use_vace_reference_image_prob,
        use_hybrid_training_with_image=args.use_hybrid_training_with_image,
        add_noise_to_image=args.add_noise_to_image,
        over_exposure_prompt_path=args.over_exposure_prompt_path,
        under_exposure_prompt_path=args.under_exposure_prompt_path,
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        srgb_to_lg=args.srgb_to_lg,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        use_swanlab=args.use_swanlab,
        log_every_n_steps=args.log_every_n_steps,
        exp_name=args.exp_name,
        save_every_n_epochs=args.save_every_n_epochs,
    )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        is_exr=args.is_exr,
        pin_memory=args.pin_memory,
        num_workers=args.dataset_num_workers,
        prefetch_factor=args.prefetch_factor,
        save_steps=args.save_steps,
        max_train_steps=args.max_train_steps,
        resume_lora=args.resume_lora,
        resume_step=args.resume_step,
        find_unused_parameters=args.find_unused_parameters,
    )
