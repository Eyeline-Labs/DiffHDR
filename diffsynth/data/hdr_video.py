# HDR video dataset: assumes --is_exr
# --use_exposured_gt --srgb_to_lg --add_exposure_mask --use_under_expose_mask
# --add_noise_to_image
# --use_exposure_augmentation --use_vace_reference_image --use_hybrid_training_with_image
# --use_exposure_prompt --lora_base_model vace

import os
import json

import torch
import torchvision
import pandas as pd
from PIL import Image
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.color_utils import (
    Lin_to_Log, Log_to_Lin,
    exposure_masks_from_srgb, stabilize_soft_mask,
    rec709_linear_to_srgb_torch, srgb_to_linear_rec709_torch,
)
from utils.noise_utils import add_noise_to_video

import pyexr


class HDRVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        repeat=1,
        args=None,
        used_data_percentage=1.0,
        prompt_path=None,
        use_exposure_augmentation=False,
        srgb_to_lg=False,
        use_exposured_gt=False,
        add_exposure_mask=False,
        add_noise_prob=0.3,
        use_under_expose_mask=False,
        use_exposure_prompt=False,
        exposure_prompt_path=None,
        over_exposure_prompt_path=None,
        under_exposure_prompt_path=None,
        use_vace_reference_image=False,
        use_vace_reference_image_prob=1.0,
        return_video_name=False,
        use_hybrid_training_with_image=False,
        add_noise_to_image=False,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.repeat = repeat
        self.prompt_path = prompt_path
        self.use_exposure_augmentation = use_exposure_augmentation
        self.srgb_to_lg = srgb_to_lg
        self.use_exposured_gt = use_exposured_gt
        self.add_exposure_mask = add_exposure_mask
        self.add_noise_prob = add_noise_prob
        self.use_under_expose_mask = use_under_expose_mask
        self.use_vace_reference_image = use_vace_reference_image
        self.use_vace_reference_image_prob = use_vace_reference_image_prob
        self.return_video_name = return_video_name
        self.use_hybrid_training_with_image = use_hybrid_training_with_image
        self.add_noise_to_image = add_noise_to_image
        self.over_exposure_prompt_path = over_exposure_prompt_path
        self.under_exposure_prompt_path = under_exposure_prompt_path

        # Height and width must be provided (no dynamic_resolution)
        assert height is not None and width is not None, \
            "height and width must be specified"

        # Load metadata (CSV or JSON)
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        if used_data_percentage < 1.0:
            self.data = self.data[:int(len(self.data) * used_data_percentage)]

        # Load prompts (qwen3-vl format)
        if self.prompt_path is not None:
            meatadata_prompt = pd.read_csv(self.prompt_path)
            self.prompt_data = {}
            for i in range(len(meatadata_prompt)):
                video_name = (meatadata_prompt.iloc[i]['video'].split('/')[-3]
                              + '_' + meatadata_prompt.iloc[i]['video'].split('/')[-2])
                self.prompt_data[video_name] = meatadata_prompt.iloc[i]['prompt']

        # Exposure prompts (always loaded when use_exposure_prompt is True)
        self.use_exposure_prompt = use_exposure_prompt
        if exposure_prompt_path is not None and self.use_exposure_prompt:
            exposure_metadata = pd.read_csv(exposure_prompt_path)
            self.exposure_prompt = {}
            for i in range(len(exposure_metadata)):
                key = f"{exposure_metadata.iloc[i]['video']}"
                self.exposure_prompt[key] = (
                    exposure_metadata.iloc[i]['text']
                    if not pd.isna(exposure_metadata.iloc[i]['text']) else ""
                )
            print(f"Loaded exposure prompts for {len(self.exposure_prompt)} video-exposure pairs.")

        if over_exposure_prompt_path is not None:
            over_exposure_metadata = pd.read_csv(over_exposure_prompt_path)
            self.over_exposure_prompt = {}
            for i in range(len(over_exposure_metadata)):
                key = f"{over_exposure_metadata.iloc[i]['video']}"
                key = key.split('/')[-3] + '_' + key.split('/')[-2]
                self.over_exposure_prompt[key] = (
                    over_exposure_metadata.iloc[i]['prompt']
                    if not pd.isna(over_exposure_metadata.iloc[i]['prompt']) else ""
                )
                if 'silhouette' in self.over_exposure_prompt[key]:
                    self.over_exposure_prompt[key] = (
                        "over-exposed: Recover the content in the over-exposed area."
                    )
            print(f"Loaded over exposure prompts for {len(self.over_exposure_prompt)} video-exposure pairs.")

        if under_exposure_prompt_path is not None:
            under_exposure_metadata = pd.read_csv(under_exposure_prompt_path)
            self.under_exposure_prompt = {}
            for i in range(len(under_exposure_metadata)):
                key = f"{under_exposure_metadata.iloc[i]['video']}"
                key = key.split('/')[-3] + '_' + key.split('/')[-2]
                self.under_exposure_prompt[key] = (
                    under_exposure_metadata.iloc[i]['prompt']
                    if not pd.isna(under_exposure_metadata.iloc[i]['prompt']) else ""
                )
                if 'silhouette' in self.under_exposure_prompt[key]:
                    self.under_exposure_prompt[key] = (
                        "under-exposed: Recover the content in the under-exposed area."
                    )
            print(f"Loaded under exposure prompts for {len(self.under_exposure_prompt)} video-exposure pairs.")

    def crop_and_resize(self, image, target_height, target_width, is_exr=False, resize=True):
        if is_exr:
            width, height = image.shape[2], image.shape[1]
        else:
            width, height = image.size
        if resize:
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height * scale), round(width * scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
        image = torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )
        return image

    def load_exr(self, file_path):
        height = self.height
        width = self.width

        frames = []
        start_frame = int(file_path.split('/')[-1][6:-4])

        if self.use_exposure_augmentation and self.use_exposured_gt:
            exposure = np.random.randint(-1, 3)
            if exposure > 3:
                exposure = 3
        else:
            exposure = 0

        # Random sampling: randomly sample consecutive num_frames from 81 frames
        if self.num_frames >= 81:
            frame_indices = list(range(81))
        else:
            max_start = 81 - self.num_frames
            start_idx = np.random.randint(0, max_start + 1)
            frame_indices = list(range(start_idx, start_idx + self.num_frames))

        for i in frame_indices:
            frame_index = start_frame + i
            frame_path = os.path.join(
                os.path.dirname(file_path),
                f"frame_{frame_index:04d}.exr",
            )

            frame = pyexr.read(frame_path)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)

            if self.use_exposure_augmentation and self.use_exposured_gt:
                frame = frame * (2.0 ** exposure)
                frame = torch.clamp(frame, min=0.0, max=65536.0)

            frames.append(frame)

        return frames, exposure

    def load_data(self, file_path):
        return self.load_exr(file_path)

    def load_srgb_from_linear(self, data, video_name, exposure=0):
        data['vace_video'] = []
        data['prompt'] = ""
        mask_over_bin_prob = np.random.rand()

        if self.use_hybrid_training_with_image:
            random_prob_with_image = np.random.rand()

        # use_hybrid_training_with_image: 1/8 chance to use original prompt
        if self.use_hybrid_training_with_image:
            if random_prob_with_image < 0.125:
                gray_frame = torch.zeros_like(data['video'][0]) + 0.5
                gray_frame = srgb_to_linear_rec709_torch(gray_frame)
                gray_frame = Lin_to_Log(gray_frame)
                data['vace_video'] = [gray_frame for _ in data['video']]

                for i in range(len(data['video'])):
                    if exposure != 0:
                        data['video'][i] = data['video'][i] * (2.0 ** (-exposure))
                    data['video'][i] = Lin_to_Log(data['video'][i])

                data['vace_video_mask'] = [
                    Image.fromarray(
                        np.ones((frame.shape[1], frame.shape[2], 3), dtype=np.uint8) * 255
                    )
                    for frame in data['video']
                ]
                data['prompt'] = self.prompt_data[video_name]
                return 0

        # Prompt assignment for use_hybrid_training_with_image
        if self.use_hybrid_training_with_image:
            if random_prob_with_image >= 0.125 and random_prob_with_image < 0.25:
                # 1/8 chance: only v/m (empty prompt)
                data['prompt'] = ""
            elif random_prob_with_image >= 0.25 and random_prob_with_image < 0.75:
                # 3/8 chance: t + v/m
                if (self.over_exposure_prompt_path is not None
                        and self.under_exposure_prompt_path is not None):
                    over_exposure_prompt = self.over_exposure_prompt[video_name]
                    under_exposure_prompt = self.under_exposure_prompt[video_name]
                    data['prompt'] = over_exposure_prompt + " " + under_exposure_prompt
                else:
                    data['prompt'] = self.prompt_data[video_name]
            else:
                # remaining: empty prompt
                data['prompt'] = ""

        if self.add_exposure_mask:
            data['vace_video_mask'] = []
            prev_over_ema = prev_under_ema = None

        # Reference image setup
        if self.use_vace_reference_image:
            use_ref = (
                (np.random.rand() < self.use_vace_reference_image_prob
                 and not self.use_hybrid_training_with_image)
                or (self.use_hybrid_training_with_image
                    and random_prob_with_image >= 0.625)
            )
            if use_ref:
                index = 0
                ref_image_lin = data['video'][index].clone()
                data['vace_reference_image'] = ref_image_lin
            else:
                data['vace_reference_image'] = None

        if self.use_under_expose_mask:
            prob_random_mask = np.random.rand()

        # Add noise to video
        if self.add_noise_to_image and np.random.rand() < self.add_noise_prob:
            noisy_video, _ = add_noise_to_video(
                data,
                white_level=65536.0,
                per_channel=True,
                temporal_ar=0.5,
            )
        else:
            noisy_video = None

        over_luma_thr_srgb = 0.95
        under_luma_thr_srgb = 0.05
        clip_eps_srgb = 0.02
        only_over_bin = False
        only_under_bin = False

        for i in range(len(data['video'])):

            if noisy_video is not None:
                frame_lin = noisy_video[i]
            else:
                frame_lin = data['video'][i].clone()

            frame_srgb = rec709_linear_to_srgb_torch(frame_lin).unsqueeze(0)
            data['video'][i] = Lin_to_Log(data['video'][i])

            if self.srgb_to_lg:
                # Simulate 8-bit quantization
                frame_srgb = (frame_srgb * 255.0).to(torch.uint8)
                frame_srgb = frame_srgb.to(torch.float32) / 255.0

                if self.add_exposure_mask:
                    over_soft, under_soft = exposure_masks_from_srgb(
                        frame_srgb.squeeze(0),
                        over_luma_thr_srgb=over_luma_thr_srgb,
                        under_luma_thr_srgb=under_luma_thr_srgb,
                        softness=0.02,
                        clip_eps_srgb=clip_eps_srgb,
                        use_local_contrast_gate=False,
                    )

                    over_soft, prev_over_ema = stabilize_soft_mask(
                        over_soft, prev_over_ema, alpha=0.7, k_smooth=9, k_open_close=9
                    )
                    under_soft, prev_under_ema = stabilize_soft_mask(
                        under_soft, prev_under_ema, alpha=0.7, k_smooth=9, k_open_close=9
                    )

                    over_bin = over_soft > 0.2
                    under_bin = under_soft > 0.2

                    combined_bin = (over_bin | under_bin).float()
                    mask = under_bin.to(frame_srgb.device).float()[None, None, :, :]
                    frame_srgb = frame_srgb * (1.0 - mask) + 0.5 * mask

                    # Randomly mask over-exposed regions to gray
                    if mask_over_bin_prob < 0.1:
                        mask = over_bin.to(frame_srgb.device).float()[None, None, :, :]
                        frame_srgb = frame_srgb * (1.0 - mask) + 0.5 * mask

                    combined_bin = combined_bin.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]
                    data['vace_video_mask'].append(
                        Image.fromarray((combined_bin * 255.0).numpy().astype(np.uint8))
                    )

                frame_srgb = srgb_to_linear_rec709_torch(frame_srgb)
                frame_srgb = Lin_to_Log(frame_srgb)
                data['vace_video'].append(frame_srgb[0])
            else:
                data['vace_video'].append(
                    Image.fromarray(
                        (frame_srgb[0] * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
                    )
                )

        return exposure

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        path = data['video']

        data['video'], exposure = self.load_data(path)

        if "vace_video" in self.data_file_keys:
            video_name = path.split('/')[-3] + '_' + path.split('/')[-2]
            self.load_srgb_from_linear(
                data,
                video_name=video_name,
                exposure=exposure,
            )
            if self.return_video_name:
                data['video_name'] = video_name
                data['exposure'] = exposure

        if ("vace_reference_image" in self.data_file_keys
                and "vace_reference_image" not in data.keys()):
            data['vace_reference_image'] = None


        return data

    def __len__(self):
        return len(self.data) * self.repeat
