import math
from typing import Dict, List, Tuple, Optional, Union

import torch


@torch.no_grad()
def add_noise_to_video(
    data: Dict,
    *,
    key: str = "video",
    white_level: float = 65536.0,
    # CBDNet-style ranges, interpreted in normalized [0,1] domain
    sigma_s_range: Tuple[float, float] = (0.0, 0.00085),
    sigma_c_range: Tuple[float, float] = (0.0, 0.000015),
    # Optionally fix params (float or tensor[C])
    sigma_s: Optional[Union[float, torch.Tensor]] = None,
    sigma_c: Optional[Union[float, torch.Tensor]] = None,
    # Sharing / consistency
    per_channel: bool = True,     # one sigma per channel (shared across frames)
    temporal_ar: float = 0.0,     # 0 -> independent per frame; 0.3~0.7 mild temporal consistency
    # Output / safety
    clamp: bool = True,
    eps: float = 1e-12,
    # Randomness control
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    # Return mode
    return_new_data: bool = False,  # False: return noisy_video, info; True: return new_data, info
    noise_strength: float = 0.01,
):
    """
    data[key] is a list of tensors: [C,H,W] each. Values in linear Rec.709, range [0, white_level].

    This function NEVER modifies data[key] in-place.
    It returns a new list (or a new dict if return_new_data=True).

    Noise model (in normalized domain):
      n ~ N(0, L*sigma_s^2 + sigma_c^2)
    Then noise is scaled back by white_level and added to x.
    """
    if key not in data:
        raise KeyError(f"data has no key '{key}'")

    video: List[torch.Tensor] = data[key]
    if not isinstance(video, list) or len(video) == 0:
        raise ValueError(f"data['{key}'] must be a non-empty list of tensors")

    # Validate shapes and collect device/dtype
    x0 = video[0]
    if not torch.is_tensor(x0) or x0.ndim != 3:
        raise ValueError(f"Expected each frame to be a torch.Tensor of shape [C,H,W], got {type(x0)} with ndim={getattr(x0,'ndim',None)}")
    C, H, W = x0.shape
    device = x0.device
    dtype = x0.dtype

    for i, x in enumerate(video):
        if (not torch.is_tensor(x)) or x.ndim != 3:
            raise ValueError(f"Frame {i} is not a tensor [C,H,W]")
        if x.shape != (C, H, W):
            raise ValueError(f"Frame {i} shape mismatch: {x.shape} vs {(C,H,W)}")
        if x.device != device:
            raise ValueError(f"Frame {i} device mismatch: {x.device} vs {device}")
        if x.dtype != dtype:
            raise ValueError(f"Frame {i} dtype mismatch: {x.dtype} vs {dtype}")

    # Sample/prepare sigma_s, sigma_c in normalized domain [0,1]
    def _as_sigma(value, name: str) -> torch.Tensor:
        if value is None:
            lo, hi = (sigma_s_range if name == "sigma_s" else sigma_c_range)
            if per_channel:
                return (lo + (hi - lo) * torch.rand((C,), device=device)).to(torch.float32)
            else:
                return torch.tensor(float(lo + (hi - lo) * torch.rand((), device=device).item()),
                                    device=device, dtype=torch.float32)
        if isinstance(value, (float, int)):
            if per_channel:
                return torch.full((C,), float(value), device=device, dtype=torch.float32)
            else:
                return torch.tensor(float(value), device=device, dtype=torch.float32)
        if torch.is_tensor(value):
            v = value.to(device=device, dtype=torch.float32)
            if v.ndim == 0:
                return torch.full((C,), float(v.item()), device=device, dtype=torch.float32) if per_channel else v
            if v.shape == (C,):
                return v if per_channel else torch.tensor(float(v.mean().item()), device=device, dtype=torch.float32)
            raise ValueError(f"{name} tensor must be scalar or shape (C,), got {tuple(v.shape)}")
        raise TypeError(f"Unsupported type for {name}: {type(value)}")

    sig_s = _as_sigma(sigma_s, "sigma_s")
    sig_c = _as_sigma(sigma_c, "sigma_c")

    # Precompute AR(1) settings (temporal consistency)
    rho = float(max(0.0, min(0.999, temporal_ar)))
    ar_scale = math.sqrt(max(1.0 - rho * rho, 0.0))

    # Create new noisy frames (no in-place modification)
    noisy_video: List[torch.Tensor] = []

    # AR state in normalized noise domain: shape [C,H,W]
    e_prev = torch.zeros((C, H, W), device=device, dtype=torch.float32) if rho > 0 else None

    # reshape sigmas for broadcasting: [C,1,1]
    if per_channel:
        sig_s2 = (sig_s ** 2).view(C, 1, 1)
        sig_c2 = (sig_c ** 2).view(C, 1, 1)
    else:
        sig_s2 = (sig_s ** 2).view(1, 1, 1)
        sig_c2 = (sig_c ** 2).view(1, 1, 1)

    for x in video:
        # Work in float32 for stable noise math; keep output dtype same as input
        x_f = x.to(torch.float32)

        # normalized irradiance L in [0, +inf), (we clip negatives to 0 for variance)
        L = (x_f / white_level).clamp_min(0.0)

        # var in normalized domain: L * sigma_s^2 + sigma_c^2
        var = L * sig_s2 + sig_c2
        std = torch.sqrt(var.clamp_min(eps))

        if rho <= 0.0:
            z = torch.randn((C, H, W), device=device, dtype=torch.float32)
        else:
            u = torch.randn((C, H, W), device=device, dtype=torch.float32)
            z = rho * e_prev + ar_scale * u
            e_prev = z

        noise = z * std * white_level * noise_strength
        y = x_f + noise

        if clamp:
            y = y.clamp(0.0, white_level)

        noisy_video.append(y.to(dtype))  # new tensor, not modifying x

    info = {
        "white_level": float(white_level),
        "sigma_s": sig_s.detach().cpu().tolist() if sig_s.ndim == 1 else float(sig_s.item()),
        "sigma_c": sig_c.detach().cpu().tolist() if sig_c.ndim == 1 else float(sig_c.item()),
        "per_channel": bool(per_channel),
        "temporal_ar": float(rho),
        "clamp": bool(clamp),
        "seed": int(seed) if seed is not None else None,
        "sigma_s_range": sigma_s_range,
        "sigma_c_range": sigma_c_range,
    }

    if return_new_data:
        # shallow copy dict so original data is untouched; replace only the video key
        new_data = dict(data)
        new_data[key] = noisy_video
        return new_data, info

    return noisy_video, info
