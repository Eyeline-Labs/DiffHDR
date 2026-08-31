import torch
import numpy as np
import math
import torch.nn.functional as F
from math import sqrt

# ACES AP0 -> AP1 matrix (ACES2065-1 -> ACEScg/ACEScct primaries).
_AP0_TO_AP1 = torch.tensor([
    [ 1.45143932, -0.23651075, -0.21492857],
    [-0.07655377,  1.17622970, -0.09967593],
    [ 0.00831615, -0.00603245,  0.99771630],
], dtype=torch.float32)

# ACES2065-1 (AP0, D60) -> linear sRGB / Rec.709 (D65)
_AP0_TO_LINEAR_SRGB = torch.tensor([
    [ 2.52168618674388, -1.13413098823972, -0.387555198504164],
    [-0.276479914229922,  1.37271908766826, -0.09623917343833401],
    [-0.0153780649660342,-0.152975335867399,  1.16835340083343],
], dtype=torch.float32)

def _to_chw(x: torch.Tensor):
    if x.ndim == 4 and x.shape[1] == 3:   # B,3,H,W
        return x, False
    if x.ndim == 3 and x.shape[0] == 3:   # 3,H,W
        return x.unsqueeze(0), False
    if x.ndim == 3 and x.shape[-1] == 3:  # H,W,3
        return x.permute(2,0,1).unsqueeze(0).contiguous(), True
    raise ValueError("Expected [B,3,H,W], [3,H,W], or [H,W,3].")

def _from_chw(x: torch.Tensor, was_hwc: bool):
    return x.permute(1,2,0).contiguous() if was_hwc else x

def apply_exposure_linear(x: torch.Tensor, ev) -> torch.Tensor:
    return x * (2.0 ** torch.as_tensor(ev, dtype=x.dtype, device=x.device))

def aces2065_to_linear_srgb_torch(x_aces: torch.Tensor, exposure_stops=0.0) -> torch.Tensor:
    """
    ACES2065-1 (AP0,D60, linear) -> linear sRGB (D65), matrix-only, GPU.
    Preserves input layout ([B,3,H,W] / [3,H,W] / [H,W,3]).
    """
    x, was_hwc = _to_chw(x_aces)             # -> [B,3,H,W]
    if exposure_stops is not None:
        x = apply_exposure_linear(x, exposure_stops)

    # Do 3x3 on channel axis via broadcasted matmul:
    # [B,H,W,3] @ [3,3]^T -> [B,H,W,3]
    M = _AP0_TO_LINEAR_SRGB.to(x.device, x.dtype)
    x = x.permute(0,2,3,1)                   # [B,H,W,3]
    x = torch.matmul(x, M.t())               # [B,H,W,3]
    x = x.permute(0,3,1,2).contiguous()      # [B,3,H,W]

    return _from_chw(x, was_hwc)

def reinhard_per_channel(x: torch.Tensor) -> torch.Tensor:
    """Simple global tonemap per channel: x / (1 + x)."""
    return x / (1.0 + torch.clamp(x, min=0.0))

def srgb_oetf(x_lin: torch.Tensor) -> torch.Tensor:
    a = 0.0031308
    low  = 12.92 * x_lin
    high = 1.055 * torch.pow(torch.clamp(x_lin, min=0.0), 1/2.4) - 0.055
    return torch.where(x_lin <= a, low, high)

def aces2065_to_srgb_torch(
    x_aces: torch.Tensor,
    exposure_stops: float | torch.Tensor = 0.0,
    tonemap: str | None = None,   # None or "reinhard"
    clamp_linear: bool = True,    # clip negatives before OETF
) -> torch.Tensor:
    """
    Full pipeline: ACES2065-1 (linear) -> linear sRGB -> (optional tonemap) -> sRGB OETF.
    Returns sRGB in [0,1], same layout as input, stays on GPU & keeps grads.
    """
    x_lin = aces2065_to_linear_srgb_torch(x_aces, exposure_stops=exposure_stops)

    if clamp_linear:
        x_lin = torch.clamp(x_lin, min=0.0)

    if tonemap is not None:
        if tonemap.lower() == "reinhard":
            x_lin = reinhard_per_channel(x_lin)
        else:
            raise ValueError(f"Unknown tonemap '{tonemap}'. Use None or 'reinhard'.")

    x_srgb = torch.clamp(srgb_oetf(x_lin), 0.0, 1.0)
    return x_srgb


def srgb_oetf_torch(x: torch.Tensor) -> torch.Tensor:
    """sRGB OETF: linear -> sRGB-encoded."""
    a = 0.055
    t = 0.0031308
    out = torch.empty_like(x)
    mask = x <= t
    out[mask] = 12.92 * x[mask]
    out[~mask] = (1.0 + a) * torch.pow(x[~mask], 1.0 / 2.4) - a
    return out

def rec709_linear_to_srgb_torch(
    x_lin709: torch.Tensor,
    exposure_stops: float | torch.Tensor = 0.0,
    clamp_linear: bool = True,   # clip negatives before OETF
    clamp_output: bool = True,   # clamp final sRGB to [0,1]
) -> torch.Tensor:
    """
    linear Rec.709 (same primaries/white as sRGB) -> sRGB-encoded, with exposure control.

    Args:
        x_lin709: linear-light RGB in Rec.709/sRGB primaries, any shape.
        exposure_stops: EV offset in stops (+1 => *2, -1 => /2). Can be float or tensor.
        clamp_linear: clamp negatives to 0 before OETF.
        clamp_output: clamp output to [0,1].
    """
    x = x_lin709

    # Apply exposure in linear domain (keeps grads)
    if isinstance(exposure_stops, (int, float)) and exposure_stops != 0.0:
        x = x * (2.0 ** float(exposure_stops))
    else:
        ev = torch.as_tensor(exposure_stops, dtype=x.dtype, device=x.device)
        if torch.any(ev != 0):
            x = x * torch.pow(torch.tensor(2.0, dtype=x.dtype, device=x.device), ev)

    if clamp_linear:
        x = torch.clamp(x, min=0.0)

    y = srgb_oetf_torch(x)

    if clamp_output:
        y = y.clamp(0.0, 1.0)

    return y

def srgb_eotf_torch(x: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF: sRGB-encoded -> linear."""
    a = 0.055
    t = 0.04045
    out = torch.empty_like(x)
    mask = x <= t
    out[mask] = x[mask] / 12.92
    out[~mask] = torch.pow((x[~mask] + a) / (1.0 + a), 2.4)
    return out


def srgb_to_linear_rec709_torch(
    x_srgb: torch.Tensor,
    exposure_stops: float | torch.Tensor = 0.0,
    clamp_input: bool = True,    # clamp sRGB to [0,1] before EOTF
    clamp_linear: bool = True,   # clamp negatives after inverse exposure (optional)
) -> torch.Tensor:
    """
    sRGB-encoded -> linear Rec.709 (same primaries/white as sRGB).
    Args:
        x_srgb: sRGB in [0,1] (any shape, stays on GPU, keeps grads)
        exposure_stops: if your sRGB was produced with +EV in linear before encoding,
                        set the same value here to undo it (divide by 2^EV).
        clamp_input: clamp x_srgb to [0,1] before EOTF (recommended).
        clamp_linear: clamp linear result to >=0 after exposure undo (recommended).
    Returns:
        linear Rec.709 RGB (float), same shape as input.
    """
    x = x_srgb
    if clamp_input:
        x = x.clamp(0.0, 1.0)

    # 1) sRGB decode: encoded -> linear
    lin = srgb_eotf_torch(x)

    # 2) Undo exposure in linear domain (inverse of the forward exposure)
    # forward: lin *= 2^EV  => here: lin /= 2^EV
    if isinstance(exposure_stops, (int, float)) and float(exposure_stops) != 0.0:
        lin = lin * (2.0 ** (-float(exposure_stops)))
    else:
        ev = torch.as_tensor(exposure_stops, dtype=lin.dtype, device=lin.device)
        if torch.any(ev != 0):
            lin = lin * torch.pow(torch.tensor(2.0, dtype=lin.dtype, device=lin.device), -ev)

    if clamp_linear:
        lin = torch.clamp(lin, min=0.0)

    return lin

# linear sRGB (D65) -> ACES2065-1 (AP0, D60)
# (inverse of your AP0->lin sRGB matrix; includes D60<->D65 via Bradford)
_LINEAR_SRGB_TO_AP0 = torch.tensor([
    [0.4396329811428639, 0.3829887025841661, 0.1773783162729699],
    [0.0897764446334716, 0.8134394311128358, 0.0967841242536926],
    [0.0175411683939797, 0.1115465500611321, 0.8709122815448882],
], dtype=torch.float32)

def srgb_eotf_torch(E: torch.Tensor) -> torch.Tensor:
    """
    Inverse sRGB OETF: encoded sRGB in [0,1] -> linear sRGB.
    Do pow in fp32 for stability, then cast back.
    """
    a = 0.04045
    Ef = E.float()
    low  = Ef / 12.92
    high = torch.pow((torch.clamp(Ef, 0.0, 1.0) + 0.055) / 1.055, 2.4)
    out  = torch.where(Ef <= a, low, high)
    return out.to(dtype=E.dtype)

def srgb_to_aces2065_torch(
    x_srgb: torch.Tensor,
    # inverse_exposure_stops: float | torch.Tensor = 0.0,  # undo prior EV applied in ACES (optional)
    clamp_input: bool = True,   # clamp sRGB to [0,1] before EOTF
    channels_last: bool = False,  # use NHWC memory format for the 1x1 conv
    is_lin: bool = False,
) -> torch.Tensor:
    """
    sRGB-encoded -> ACES2065-1 (scene-linear, AP0/D60).
    Accepts [B,3,H,W], [3,H,W], or [H,W,3]; returns same layout (no squeeze).
    """
    x, was_hwc = _to_chw(x_srgb)  # -> [B,3,H,W]

    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    # 1) inverse OETF: sRGB -> linear sRGB
    if clamp_input and not is_lin:
        x = x.clamp(0.0, 1.0)
    if is_lin:
        lin_srgb = x
    else:
        lin_srgb = srgb_eotf_torch(x)

    # 2) 1x1 conv: linear sRGB -> ACES2065-1
    W = _LINEAR_SRGB_TO_AP0.view(3,3,1,1).to(lin_srgb.device, lin_srgb.dtype)
    if channels_last:
        W = W.to(memory_format=torch.channels_last)
    aces = F.conv2d(lin_srgb, W, bias=None, stride=1, padding=0)

    return _from_chw(aces, was_hwc)


### Exposure masks from sRGB images ###
def _linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92*x, 1.055*torch.clamp(x,0,1).pow(1/2.4) - 0.055)

def _srgb_to_linear(x: torch.Tensor):
    # x: [..., H, W] or [3, H, W], in [0,1]
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _luma709_linear(rgb_lin: torch.Tensor):
    # rgb_lin: [3,H,W]
    return 0.2126 * rgb_lin[0] + 0.7152 * rgb_lin[1] + 0.0722 * rgb_lin[2]

def _smoothstep(x, edge0, edge1):
    # returns 0..1
    t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1)
    return t * t * (3 - 2 * t)

def _low_contrast_mask(gray: torch.Tensor, ksize: int = 3, thr: float = 0.03):
    # crude local contrast gate using max-min in k×k neighborhood (linear domain)
    pad = ksize // 2
    g = gray.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    gmax = F.max_pool2d(g, ksize, stride=1, padding=pad)
    gmin = -F.max_pool2d(-g, ksize, stride=1, padding=pad)
    lc = (gmax - gmin).squeeze(0).squeeze(0)
    # return 1 where contrast is LOW (i.e., likely flat/clipped region)
    return (lc <= thr).float()

@torch.no_grad()
def exposure_masks_from_srgb(
    rgb_srgb_chw: torch.Tensor,
    over_luma_thr_srgb: float = 0.95,#0.98,
    under_luma_thr_srgb: float = 0.01,
    softness: float = 0.02,           # soft band for smoothstep in sRGB
    clip_eps_srgb: float = 0.01,      # channel clip tolerance in sRGB
    use_local_contrast_gate: bool = False,
    lc_ksize: int = 3,
    lc_thr_linear: float = 0.07,
    exposure_stops: float = 0.0,
):
    """
    Inputs:
        rgb_srgb_chw: [3,H,W], float32 in [0,1] sRGB (gamma-coded)
    Returns:
        over_mask, under_mask: [H,W] soft masks in [0,1] (torch.float32)
    """
    device = rgb_srgb_chw.device
    # linearize for robust luma and contrast computation
    rgb_lin = _srgb_to_linear(rgb_srgb_chw.clamp(0, 1))
    if exposure_stops != 0.0:
        rgb_lin = rgb_lin * (2.0 ** float(exposure_stops))
        re_srgb_chw = rec709_linear_to_srgb_torch(rgb_lin)
    Y = _luma709_linear(rgb_lin)
    #top2 = torch.topk(rgb_lin, k=2, dim=0).values        # [2,H,W]
    #Y = top2.mean(dim=0)                                 # [H,W]

    # --- Overexposure (soft) ---
    # 1) luma near top in sRGB (soft range)
    over_l = _smoothstep(
        x=rgb_srgb_chw.new_tensor(Y), 
        edge0=_srgb_to_linear(torch.tensor(over_luma_thr_srgb - softness, device=device)),
        edge1=_srgb_to_linear(torch.tensor(over_luma_thr_srgb + softness, device=device)),
    )
    # 2) channel clipping: at least 2 channels near 1.0 in sRGB
    near1 = (rgb_srgb_chw >= (1.0 - clip_eps_srgb)).float()
    ch_clip = (near1.sum(dim=0) >= 2).float()

    over = (0.7 * over_l + 0.3 * ch_clip).clamp(0, 1)

    # optional: suppress bright textured edges/speculars using low-contrast gate
    if use_local_contrast_gate:
        lc_gate = _low_contrast_mask(Y, ksize=lc_ksize, thr=lc_thr_linear)  # 1 where flat
        over = (over * lc_gate).clamp(0, 1)

    # --- Underexposure (soft) ---
    # Soft threshold in linear domain is more meaningful in shadows.
    under = 1.0 - _smoothstep(
        x=Y,
        edge0=_srgb_to_linear(torch.tensor(max(0, under_luma_thr_srgb - softness), device=device)),
        edge1=_srgb_to_linear(torch.tensor(under_luma_thr_srgb + softness, device=device)),
    )
    
    # also require all channels to be low-ish in sRGB to avoid colored dark edges
    all_low = (rgb_srgb_chw <= under_luma_thr_srgb).all(dim=0).float()
    under = (0.7 * under + 0.3 * all_low).clamp(0, 1)

    # small morphological smooth (optional): 3x3 closing to fill pinholes
    def _morph_smooth(m):
        m4 = m.unsqueeze(0).unsqueeze(0)
        m4 = F.max_pool2d(m4, 3, stride=1, padding=1)  # dilate
        m4 = -F.max_pool2d(-m4, 3, stride=1, padding=1)  # erode
        return m4.squeeze(0).squeeze(0)

    over = _morph_smooth(over)
    under = _morph_smooth(under)

    if exposure_stops != 0.0:
        return re_srgb_chw, over.clamp(0, 1), under.clamp(0, 1)
    else:
        return over.clamp(0, 1), under.clamp(0, 1)


@torch.no_grad()
def binarize_soft_mask(
    soft: torch.Tensor,               # [H,W] in [0,1]
    prev_ema: torch.Tensor | None,    # [H,W] or None
    prev_bin: torch.Tensor | None,    # [H,W] or None (0/1)
    alpha: float = 0.7,               # EMA coeff (higher = smoother)
    tau_on: float = 0.60,             # turn-on threshold
    tau_off: float = 0.45,            # turn-off threshold (tau_off < tau_on)
    k_smooth: int = 7,                # light spatial smoothing
    k_open_close: int = 7             # small open/close for speckle/holes
):
    H, W = soft.shape
    # 1) temporal EMA
    ema = soft if prev_ema is None else (alpha * soft + (1 - alpha) * prev_ema)

    # 2) light spatial blur before threshold (reduces single-pixel noise)
    if k_smooth > 1:
        p = k_smooth // 2
        ema_b = F.avg_pool2d(ema[None,None], k_smooth, 1, p)[0,0]
    else:
        ema_b = ema

    # 3) hysteresis thresholding
    if prev_bin is None:
        bin_mask = (ema_b >= tau_on).float()
    else:
        stay_on  = (prev_bin > 0.5) & (ema_b >= tau_off)
        turn_on  = (prev_bin <= 0.5) & (ema_b >= tau_on)
        bin_mask = (stay_on | turn_on).float()

    # 4) tiny morphological open+close to clean specks and fill pinholes
    def erode(b, k):
        # min-pool via max-pool on negative
        b4 = b[None,None]
        e = -F.max_pool2d(-b4, k, 1, k//2)
        return e[0,0]
    def dilate(b, k):
        b4 = b[None,None]
        d = F.max_pool2d(b4, k, 1, k//2)
        return d[0,0]

    if k_open_close > 1:
        # opening (erode->dilate) removes tiny bright specks
        bin_mask = dilate(erode(bin_mask, k_open_close), k_open_close)
        # closing (dilate->erode) fills tiny holes
        bin_mask = erode(dilate(bin_mask, k_open_close), k_open_close)

    return bin_mask.clamp(0,1), ema_b


@torch.no_grad()
def stabilize_soft_mask(
    soft: torch.Tensor,               # [H,W], float32 in [0,1]
    prev_ema: torch.Tensor | None,    # [H,W] or None
    alpha: float = 0.7,               # EMA coeff (higher = steadier over time)
    k_smooth: int = 7,                # avg blur kernel for denoise (odd)
    k_open_close: int = 5                  # soft closing to fill pinholes (odd; 0/1 disables)
):
    # 1) temporal EMA
    ema = soft if prev_ema is None else (alpha * soft + (1 - alpha) * prev_ema)
    ema_state = ema.clone()

    # 2) spatial denoise (avg blur)
    if k_smooth > 1:
        p = k_smooth // 2
        ema = F.avg_pool2d(ema[None,None], k_smooth, 1, p)[0,0]

    # 4) tiny morphological open+close to clean specks and fill pinholes
    def erode(b, k):
        # min-pool via max-pool on negative
        b4 = b[None,None]
        e = -F.max_pool2d(-b4, k, 1, k//2)
        return e[0,0]
    def dilate(b, k):
        b4 = b[None,None]
        d = F.max_pool2d(b4, k, 1, k//2)
        return d[0,0]

    if k_open_close > 1:
        # opening (erode->dilate) removes tiny bright specks
        ema = dilate(erode(ema, k_open_close), k_open_close)
        # closing (dilate->erode) fills tiny holes
        ema = erode(dilate(ema, k_open_close), k_open_close)

    return ema.clamp(0, 1), ema_state

def mask_pyramid(m_soft: torch.Tensor, strides=(2,4,8,16)):
    pyr = []
    cur = m_soft[None,None]  # [1,1,H,W]
    for s in strides:
        cur = F.avg_pool2d(cur, s, s)  # anti-aliased downsample
        pyr.append(cur[0,0])
    return pyr


def blur_mask(mask, k=5):
    if k <= 1: return mask
    p = k//2
    return F.avg_pool2d(mask[None,None], k, 1, p)[0,0]

def _highpass(eps, k=7):
    """Remove low frequencies so noise reads as grain, not blotches."""
    if k <= 1: return eps
    p = k//2
    low = F.avg_pool2d(eps, k, 1, p)
    hp = eps - low
    std = hp.std(unbiased=False)
    return hp / (std + 1e-8)

@torch.no_grad()
def add_shadow_noise_srgb(
    rgb_srgb_chw: torch.Tensor,   # [3,H,W] in [0,1]
    under_soft: torch.Tensor,     # [H,W] in [0,1]
    shot: float = 1.5e-3,         # variance ~ a*x + b   (linear domain)
    read: float = 1e-5,
    mask_gamma: float = 0.7,
    mask_blur_ksize: int = 5,
    mode: str = "opponent",       # "opponent" | "hue_preserve" | "luma_only" | "rgb_equal"
    chroma_ratio: float = 0.35,   # opponent: chroma std = chroma_ratio * luminance std
    highpass_ksize: int = 7,      # 5–9 gives nice fine grain
    rng: torch.Generator | None = None,
):
    """
    Color-neutral shadow noise:
      - Build variance in linear light.
      - Gate by soft under-exposure mask.
      - Shape spectrum with high-pass so it reads like grain.
      - MODE controls how noise distributes across color:
          * 'opponent'     : mostly luminance, little chroma (natural)
          * 'hue_preserve' : noise along pixel's RGB direction (brightness-only)
          * 'luma_only'    : pure luminance noise (no color change)
          * 'rgb_equal'    : equal in R,G,B (baseline)
    """
    dev = rgb_srgb_chw.device
    H, W = under_soft.shape
    lin = _srgb_to_linear(rgb_srgb_chw.clamp(0,1))

    # Per-pixel base std from intensity (use luma so color stays neutral)
    Y = (0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2]).clamp(0,1)
    base_var = shot * Y + read
    base_std = torch.sqrt(base_var + 1e-12)  # [H,W]

    # Soft, feathered mask
    m = blur_mask(under_soft.clamp(0,1).pow(mask_gamma), k=mask_blur_ksize)  # [H,W]

    # White noise -> high-pass to avoid blotchy chroma
    g = rng if rng is not None else None
    eps = torch.randn((3,H,W), device=dev, generator=g)
    eps = _highpass(eps[None], k=highpass_ksize)[0]  # keep unit std approx

    if mode == "rgb_equal":
        # Same std in all channels
        noise_lin = (eps * base_std.unsqueeze(0) * m.unsqueeze(0))

    elif mode == "luma_only":
        # Convert to (luma, chroma1, chroma2), add only to luma, back to RGB
        # Orthonormal opponent basis (rows are orthonormal):
        # O1 ~ luminance, O2/O3 ~ chroma
        M = torch.tensor([[1/sqrt(3),  1/sqrt(3),  1/sqrt(3)],
                          [1/sqrt(2),  0.0,      -1/sqrt(2)],
                          [1/sqrt(6), -2/sqrt(6), 1/sqrt(6)]],
                         device=dev, dtype=lin.dtype)
        # Project noise to opponent
        eps_O = torch.einsum('ij,jhw->ihw', M, eps)
        eps_O[1:] = 0.0  # zero chroma noise
        eps_rgb = torch.einsum('ji,ihw->jhw', M, eps_O)
        noise_lin = eps_rgb * base_std.unsqueeze(0) * m.unsqueeze(0)

    elif mode == "hue_preserve":
        # Add brightness-only noise along the pixel's own RGB direction (preserves hue)
        v = lin / (lin.norm(dim=0, keepdim=True) + 1e-8)  # 3xHxW, unit direction
        # Fallback near black to neutral gray direction
        v = torch.where((v.abs().sum(0, keepdim=True) < 1e-6), 
                        torch.full_like(v, 1/sqrt(3)), v)
        epsY = eps[0]  # any channel; already white & high-passed
        amp = (base_std * m)[None, ...]
        noise_lin = v * (epsY * amp)

    else:  # "opponent" (recommended)
        M = torch.tensor([[1/sqrt(3),  1/sqrt(3),  1/sqrt(3)],
                          [1/sqrt(2),  0.0,      -1/sqrt(2)],
                          [1/sqrt(6), -2/sqrt(6), 1/sqrt(6)]],
                         device=dev, dtype=lin.dtype)
        eps_O = torch.einsum('ij,jhw->ihw', M, eps)  # to opponent space
        # Scale luma vs chroma
        scale_O = torch.stack([
            base_std,                               # luminance
            chroma_ratio * base_std,                # chroma 1
            chroma_ratio * base_std,                # chroma 2
        ]) * m  # gate with mask
        n_O = eps_O * scale_O
        noise_lin = torch.einsum('ji,ihw->jhw', M, n_O)  # back to RGB

    lin_noisy = torch.clamp(lin + noise_lin, 0.0, 1.0)
    return torch.clamp(_linear_to_srgb(lin_noisy), 0.0, 1.0)




def _make_base_grid(H, W, device, dtype):
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [H,W,2]

def _warp_bilinear(x, flow_xy): 
    """
    x:    [C,H,W]
    flow: [2,H,W] in pixel units, (dx, dy), maps t-1 -> t
    """
    C, H, W = x.shape
    # normalize flow to [-1,1] grid units
    fx = flow_xy[0] * (2.0 / max(W-1,1))
    fy = flow_xy[1] * (2.0 / max(H-1,1))
    grid0 = _make_base_grid(H, W, x.device, x.dtype)           # [H,W,2] (x,y) in [-1,1]
    grid = torch.empty_like(grid0)
    grid[..., 0] = grid0[..., 0] + fx
    grid[..., 1] = grid0[..., 1] + fy
    x4 = x.unsqueeze(0)                                        # [1,C,H,W]
    y = F.grid_sample(x4, grid.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=True)
    return y[0]                                                # [C,H,W]

@torch.no_grad()
def add_shadow_noise_srgb_local_temporal(
    rgb_srgb_chw: torch.Tensor,      # [3,H,W] in [0,1]
    under_soft: torch.Tensor,        # [H,W] soft shadow mask
    flow_tminus1_to_t: torch.Tensor | None,  # [2,H,W] in pixels, optional
    state: dict | None = None,       # {'eps': [3,H,W], 'std': [H,W]}
    rho: float = 0.95,               # temporal persistence of noise
    # reuse your local-aware knobs
    shot: float = 1.5e-3, read: float = 1e-5,
    mask_gamma: float = 0.7, mask_blur_ksize: int = 5,
    mode: str = "opponent", chroma_ratio: float = 0.35,
    highpass_ksize: int = 7,
    k_local: int = 7, flat_lo: float = 0.002, flat_hi: float = 0.02, w_flat: float = 0.8,
    grad_lo: float = 0.02, grad_hi: float = 0.10, w_edge: float = 0.7,
    # temporal amplitude smoothing (avoid pumping as std_map changes)
    std_ema: float = 0.8,            # 0=no smoothing; 0.8 is a good start
    occlusion: torch.Tensor | None = None,  # [H,W] in [0,1], 1=visible-from-prev
    rng: torch.Generator | None = None,
):
    """
    Returns: srgb_noisy [3,H,W] in [0,1], next_state dict
    """
    dev = rgb_srgb_chw.device
    H, W = under_soft.shape

    # --- Build std_map exactly like your local-aware version ---
    lin = _srgb_to_linear(rgb_srgb_chw.clamp(0,1))
    Y = (0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2]).clamp(0,1)
    base_std = torch.sqrt(shot * Y + read + 1e-12)

    m = blur_mask(under_soft.clamp(0,1).pow(mask_gamma), k=mask_blur_ksize)

    # local flatness (std in linear luminance)
    def _local_std_gray(Y, k=7):
        p = k//2
        mean = F.avg_pool2d(Y[None,None], k, 1, p)[0,0]
        mean2 = F.avg_pool2d((Y*Y)[None,None], k, 1, p)[0,0]
        var = (mean2 - mean*mean).clamp_min(0)
        return var.sqrt()
    def _sobel_grad_mag(Y):
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=Y.dtype, device=Y.device).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=Y.dtype, device=Y.device).unsqueeze(0).unsqueeze(0)
        gx = F.conv2d(Y[None,None], kx, padding=1)[0,0]
        gy = F.conv2d(Y[None,None], ky, padding=1)[0,0]
        return (gx*gx + gy*gy).sqrt()
    def _smoothstep01(x, a, b):
        t = ((x - a) / (b - a)).clamp(0, 1)
        return t*t*(3 - 2*t)

    Lstd = _local_std_gray(Y, k=k_local)
    flatness = 1.0 - _smoothstep01(Lstd, flat_lo, flat_hi)
    flat_gain = (1 - w_flat) + w_flat * flatness

    G = _sobel_grad_mag(Y)
    edge_free = 1.0 - _smoothstep01(G, grad_lo, grad_hi)
    edge_gain = (1 - w_edge) + w_edge * edge_free

    std_map = (base_std * m * flat_gain * edge_gain).clamp_min(0)

    # --- Temporal amplitude smoothing (optional) ---
    if state is not None and 'std' in state and flow_tminus1_to_t is not None and std_ema > 0:
        std_prev_warp = _warp_bilinear(state['std'].unsqueeze(0), flow_tminus1_to_t)[0]
        std_map = (std_ema * std_map + (1 - std_ema) * std_prev_warp).clamp_min(0)

    # --- AR(1) noise state with flow warp ---
    # innovation noise (unit variance), high-pass it so it reads like grain
    eps_new = torch.randn((3,H,W), device=dev, generator=rng)
    eps_new = _highpass(eps_new[None], k=highpass_ksize)[0]  # unit-ish std

    if state is None or 'eps' not in state:
        eps_prev = torch.zeros_like(eps_new)  # cold start: no history
    else:
        eps_prev = state['eps']

    if flow_tminus1_to_t is not None:
        eps_prev_warp = _warp_bilinear(eps_prev, flow_tminus1_to_t)
        std_prev_warp = _warp_bilinear(state['std'].unsqueeze(0), flow_tminus1_to_t)[0] if state is not None and 'std' in state else None
    else:
        eps_prev_warp = eps_prev
        std_prev_warp = state['std'] if state is not None and 'std' in state else None

    # Occlusion-aware persistence (optional): lower rho where prev content isn't visible
    if occlusion is not None:
        rho_map = (rho * occlusion.clamp(0,1)).to(eps_new.dtype)
    else:
        rho_map = torch.full((H,W), rho, device=dev, dtype=eps_new.dtype)

    # Combine warped history with innovation; keep unit variance
    innov_scale = torch.sqrt((1.0 - rho_map*rho_map).clamp_min(1e-6))
    eps_t = (rho_map.unsqueeze(0) * eps_prev_warp) + (innov_scale.unsqueeze(0) * eps_new)

    # --- Scale noise by current std_map and distribute color as before ---
    if mode == "rgb_equal":
        noise_lin = eps_t * std_map.unsqueeze(0)

    elif mode == "luma_only":
        M = torch.tensor([[1/sqrt(3),  1/sqrt(3),  1/sqrt(3)],
                          [1/sqrt(2),  0.0,      -1/sqrt(2)],
                          [1/sqrt(6), -2/sqrt(6), 1/sqrt(6)]],
                         device=dev, dtype=lin.dtype)
        eps_O = torch.einsum('ij,jhw->ihw', M, eps_t)
        eps_O[1:] = 0.0
        eps_rgb = torch.einsum('ji,ihw->jhw', M, eps_O)
        noise_lin = eps_rgb * std_map.unsqueeze(0)

    elif mode == "hue_preserve":
        v = lin / (lin.norm(dim=0, keepdim=True) + 1e-8)
        v = torch.where((v.abs().sum(0, keepdim=True) < 1e-6),
                        torch.full_like(v, 1/sqrt(3)), v)
        epsY = eps_t[0]
        noise_lin = v * (epsY * std_map)

    else:  # "opponent"
        M = torch.tensor([[1/sqrt(3),  1/sqrt(3),  1/sqrt(3)],
                          [1/sqrt(2),  0.0,      -1/sqrt(2)],
                          [1/sqrt(6), -2/sqrt(6), 1/sqrt(6)]],
                         device=dev, dtype=lin.dtype)
        eps_O = torch.einsum('ij,jhw->ihw', M, eps_t)
        scale_O = torch.stack([std_map, chroma_ratio*std_map, chroma_ratio*std_map])
        n_O = eps_O * scale_O
        noise_lin = torch.einsum('ji,ihw->jhw', M, n_O)

    out_lin = (lin + noise_lin).clamp(0,1)
    out_srgb = _linear_to_srgb(out_lin).clamp(0,1)

    next_state = {'eps': eps_t.detach(), 'std': std_map.detach()}
    return out_srgb, next_state

def make_rng(seed: int, device: torch.device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def luma_aces2065_1(x: torch.Tensor, clamp_nonneg: bool = True) -> torch.Tensor:
    """
    Compute CIE Y (relative luminance) from linear ACES2065-1 (AP0) RGB.

    Args:
        x: Tensor shaped [B, 3, T, H, W] (or [B, 3, H, W] / [B, 3, N]).
           Must be LINEAR ACES2065-1 (AP0), not ACEScg or sRGB.
        clamp_nonneg: If True, clamp Y to >= 0 to avoid negative tails from the AP0 blue primary.

    Returns:
        Tensor of Y with same batch/temporal/spatial dims as x but channel=1.
    """
    # ACES2065-1 (AP0) -> XYZ (D60) matrix Y row:
    # Y = 0.3439664498 * R + 0.7281660966 * G - 0.0721325464 * B
    Y = (
        0.3439664498 * x[:, 0:1, ...] +
        0.7281660966 * x[:, 1:2, ...] +
       -0.0721325464 * x[:, 2:2+1, ...]
    )
    if clamp_nonneg:
        Y = torch.maximum(Y, torch.zeros(1, dtype=Y.dtype, device=Y.device))
    return Y

#### Color management ####
def sRGB_to_Lin(im):
    """Convert sRGB to linear."""
    if isinstance(im, torch.Tensor):
        return torch.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    else:
        return np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)

def Lin_to_sRGB(im):
    """Convert linear to sRGB."""
    if isinstance(im, torch.Tensor):
        linear_part = 12.92 * im
        gamma_part = im.pow(1.0 / 2.4).mul_(1.055).sub_(0.055)
        return torch.where(im <= 0.0031308, linear_part, gamma_part)
    else:
        linear_part = 12.92 * im
        gamma_part = np.pow(im, 1.0 / 2.4) * 1.055 - 0.055
        return np.where(im <= 0.0031308, linear_part, gamma_part)

def Lin_to_Log(im, max_val=65536.0, gamma=2.2): #65536.0
    im = torch.clamp(im, max=max_val)
    im = torch.log(gamma * im + 1.0) / math.log(gamma * max_val + 1.0)
    return torch.pow(im, 1.0 / gamma)

def Log_to_Lin(im, max_val=65536.0, gamma=2.2): #65536.0
    im = torch.pow(im, gamma)
    im = torch.exp(im * math.log(gamma * max_val + 1.0)) - 1.0
    return im / gamma


# ---------------------------------------------------------------------------
# ACEScct <-> linear AP1 <-> ACES2065-1 transforms
# Self-contained ACEScct colour transforms
# ---------------------------------------------------------------------------

_ACESCCT_X_BRK = 0.0078125
_ACESCCT_Y_BRK = 0.155251141552511
_ACESCCT_A = 10.5402377416545
_ACESCCT_B = 0.0729055341958355

# AP1 (ACEScg) -> AP0 (ACES2065-1)
_AP1_TO_AP0 = torch.tensor([
    [ 0.6954522414, 0.1406786965, 0.1638690622],
    [ 0.0447945634, 0.8596711185, 0.0955343182],
    [-0.0055258826, 0.0040252103, 1.0015006723],
], dtype=torch.float32)


def acescct_to_linear_ap1(acescct: torch.Tensor) -> torch.Tensor:
    """ACEScct -> linear AP1. Input shape: (B,C,H,W) or (C,H,W)."""
    squeeze = acescct.ndim == 3
    if squeeze:
        acescct = acescct.unsqueeze(0)
    B, C, H, W = acescct.shape
    flat = acescct.reshape(B * C, H, W)
    scaled = flat * 17.52 - 9.72
    scaled = torch.clamp(scaled, min=-126, max=127)
    lin = torch.where(flat > _ACESCCT_Y_BRK, 2.0 ** scaled, (flat - _ACESCCT_B) / _ACESCCT_A)
    lin = lin.reshape(B, C, H, W)
    return lin.squeeze(0) if squeeze else lin


def linear_ap1_to_acescct(lin_ap1: torch.Tensor) -> torch.Tensor:
    """Linear AP1 -> ACEScct. Input shape: (B,C,H,W) or (C,H,W)."""
    squeeze = lin_ap1.ndim == 3
    if squeeze:
        lin_ap1 = lin_ap1.unsqueeze(0)
    B, C, H, W = lin_ap1.shape
    flat = lin_ap1.reshape(B * C, H, W)
    cct = torch.where(
        flat <= _ACESCCT_X_BRK,
        _ACESCCT_A * flat + _ACESCCT_B,
        (torch.log2(flat) + 9.72) / 17.52,
    )
    cct = cct.reshape(B, C, H, W)
    return cct.squeeze(0) if squeeze else cct


def _matmul_3x3(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 colour matrix to (B,3,H,W) tensor."""
    m = mat.to(device=x.device, dtype=x.dtype)
    return torch.einsum("ij,bjhw->bihw", m, x)


def acescct_to_aces2065(acescct: torch.Tensor) -> torch.Tensor:
    """ACEScct -> ACES2065-1 (linear AP0). Input: (B,3,H,W) or (3,H,W)."""
    squeeze = acescct.ndim == 3
    if squeeze:
        acescct = acescct.unsqueeze(0)
    lin_ap1 = acescct_to_linear_ap1(acescct)
    result = _matmul_3x3(lin_ap1, _AP1_TO_AP0)
    return result.squeeze(0) if squeeze else result


def aces2065_to_acescct(lin_ap0: torch.Tensor) -> torch.Tensor:
    """ACES2065-1 (linear AP0) -> ACEScct. Input: (B,3,H,W) or (3,H,W)."""
    squeeze = lin_ap0.ndim == 3
    if squeeze:
        lin_ap0 = lin_ap0.unsqueeze(0)
    lin_ap1 = _matmul_3x3(lin_ap0, _AP0_TO_AP1)
    result = linear_ap1_to_acescct(lin_ap1)
    return result.squeeze(0) if squeeze else result


class ColourUtils:
    """ACEScct colour utility. Only implements the methods used by the pipeline."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device) if isinstance(device, str) else device

    def Acescct_to_ACES2065_1(self, acescct: torch.Tensor) -> torch.Tensor:
        return acescct_to_aces2065(acescct.to(self.device))

    def ACES2065_1_to_Acescct(self, lin_ap0: torch.Tensor) -> torch.Tensor:
        return aces2065_to_acescct(lin_ap0.to(self.device))


