import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.color_utils import _LINEAR_SRGB_TO_AP0, _AP0_TO_AP1, _to_chw, _from_chw


class WanVideoColourTransformation(nn.Module):
    def __init__(
        self,
        init_W: torch.Tensor | None = None,
        init_a_lin: float = 10.5402377,
        init_b_lin: float = 0.072905534,
        init_c_log: float = 9.72,
        init_d_log: float = 17.52,
        init_threshold: float = 0.0078125,
        k: float = 1000.0,          # kept for API compatibility, unused now
        eps: float = 1e-10,
        learnable_log: bool = False,
    ):
        """
        Colour transform with:
          - trainable W (linear sRGB -> AP1)
          - fixed, strict ACES-like piecewise curve:

              if x <= T:
                  y = a_lin * x + b_lin
              else:
                  y = (log2(x) + c_log) / d_log

          with exact inverse.
        """
        super().__init__()

        # Default W: linear sRGB -> ACES AP1 (AP0->AP1 @ sRGB->AP0)
        if init_W is None:
            init_W = _AP0_TO_AP1 @ _LINEAR_SRGB_TO_AP0
        init_W = init_W.to(dtype=torch.float32)

        # --- ONLY trainable part: W ---
        # Trainable logits for W, made positive by row-wise softmax.
        self.W_logits = nn.Parameter(init_W.clone())

        if learnable_log:
            self.a_lin = nn.Parameter(torch.tensor(float(init_a_lin), dtype=torch.float32))
            self.b_lin = nn.Parameter(torch.tensor(float(init_b_lin), dtype=torch.float32))
            self.c_log = nn.Parameter(torch.tensor(float(init_c_log), dtype=torch.float32))
            self.d_log = nn.Parameter(torch.tensor(float(init_d_log), dtype=torch.float32))
            self.register_buffer("threshold", torch.tensor(float(init_threshold), dtype=torch.float32))
        else:
            self.register_buffer("a_lin", torch.tensor(float(init_a_lin), dtype=torch.float32))
            self.register_buffer("b_lin", torch.tensor(float(init_b_lin), dtype=torch.float32))
            self.register_buffer("c_log", torch.tensor(float(init_c_log), dtype=torch.float32))
            self.register_buffer("d_log", torch.tensor(float(init_d_log), dtype=torch.float32))
            self.register_buffer("threshold", torch.tensor(float(init_threshold), dtype=torch.float32))
        self.eps = float(eps)

    # ------------------------------------------------------------------
    #  Matrix helpers
    # ------------------------------------------------------------------
    def _get_W(self, device, dtype):
        """Row-wise softmax ⇒ each row positive & sums to 1."""
        W = F.softmax(self.W_logits, dim=1)
        return W.to(device=device, dtype=dtype)   # [3,3]

    def _get_inverse_W(self, device, dtype):
        # Always compute inverse in float32 for numerical stability
        W_32 = self._get_W(device, torch.float32)          # [3,3] in fp32
        inverse_W_32 = torch.linalg.inv(W_32)              # invert in fp32
        return inverse_W_32.to(device=device, dtype=dtype) # cast to requested dtype

    # ------------------------------------------------------------------
    #  Piecewise ACEScct-like curve (strict threshold)
    # ------------------------------------------------------------------
    def _linear_to_log(self, x: torch.Tensor) -> torch.Tensor:
        """
        Strict piecewise curve:

            if x <= T:
                y = a_lin * x + b_lin
            else:
                y = (log2(x) + c_log) / d_log

        where:
            T       = self.threshold
            a_lin   = self.a_lin
            b_lin   = self.b_lin
            c_log   = self.c_log
            d_log   = self.d_log
        """
        # Make sure constants are same dtype/device as x.
        a = self.a_lin.to(dtype=x.dtype, device=x.device)
        b = self.b_lin.to(dtype=x.dtype, device=x.device)
        c = self.c_log.to(dtype=x.dtype, device=x.device)
        d = self.d_log.to(dtype=x.dtype, device=x.device)
        T = self.threshold.to(dtype=x.dtype, device=x.device)

        y = torch.empty_like(x)

        mask = x <= T

        # Linear branch
        y[mask] = a * x[mask] + b

        # Log branch (clamp to avoid log(0)).
        x_log = torch.clamp(x[~mask], min=self.eps)
        y[~mask] = (torch.log2(x_log) + c) / d

        return y

    def _log_to_linear(self, y: torch.Tensor) -> torch.Tensor:
        """
        Exact inverse of _linear_to_log.

        Recall:

            T  = threshold
            yT = a_lin * T + b_lin

            if x <= T:
                y = a_lin * x + b_lin
              => x = (y - b_lin) / a_lin

            else:
                y = (log2(x) + c_log) / d_log
              => log2(x) = d_log * y - c_log
              => x = 2 ** (d_log * y - c_log)
        """
        a = self.a_lin.to(dtype=y.dtype, device=y.device)
        b = self.b_lin.to(dtype=y.dtype, device=y.device)
        c = self.c_log.to(dtype=y.dtype, device=y.device)
        d = self.d_log.to(dtype=y.dtype, device=y.device)
        T = self.threshold.to(dtype=y.dtype, device=y.device)

        # y at the threshold
        yT = a * T + b

        x = torch.empty_like(y)
        mask = y <= yT

        # Invert linear branch
        x[mask] = (y[mask] - b) / a

        # Invert log branch
        x[~mask] = torch.pow(2.0, d * y[~mask] - c)

        # Clamp to non-negative for safety
        x = torch.clamp(x, min=0.0)
        return x

    def forward(self,
        x_srgb: torch.Tensor = None,
        x_log: torch.Tensor = None,
        channels_last: bool = False,
        with_W: bool = True,
        forward_transform: bool = True,
        ):
        if forward_transform:
            return self._forward_transform(x_srgb, channels_last, with_W)
        else:
            return self._inverse_transform(x_log, channels_last, with_W)

    def _forward_transform(
        self,
        x_srgb,
        channels_last: bool = False,
        with_W: bool = True,
    ):
        """
        sRGB (encoded or linear) -> log space (fixed ACEScct-like curve).

        Steps:
          1) sRGB-encoded -> linear sRGB (if needed)
          2) linear sRGB -> AP1 via W (1x1 conv)
          3) AP1 linear -> log domain via strict piecewise curve.
        """
        x, was_hwc = _to_chw(x_srgb)

        if channels_last:
            x = x.to(memory_format=torch.channels_last)

        lin_srgb = x

        # Apply trainable 3x3 transform W (if enabled)
        if with_W:
            W = self._get_W(lin_srgb.device, lin_srgb.dtype).view(3, 3, 1, 1)
            if channels_last:
                W = W.to(memory_format=torch.channels_last)
            lin_transformed = F.conv2d(lin_srgb, W, bias=None, stride=1, padding=0)
        else:
            lin_transformed = lin_srgb

        log_transformed = self._linear_to_log(lin_transformed)
        return _from_chw(log_transformed, was_hwc)

    def _inverse_transform(
        self,
        y_log,
        channels_last: bool = False,
        with_W: bool = True,
    ):
        """
        Inverse transform:
            log space -> linear AP1 -> (optional) linear sRGB -> (optional) sRGB-encoded.

        Args:
            y_log:           tensor in the same log space produced by forward().
            to_srgb_encoded: if True, apply sRGB gamma at the end.
            with_W:          if True, undo W via W^{-1}; otherwise just invert curve.
            clamp_output:    clamp final linear sRGB to [0,1] before encoding.
        """
        y, was_hwc = _to_chw(y_log)

        if channels_last:
            y = y.to(memory_format=torch.channels_last)

        # 1) log -> linear AP1
        lin_ap1 = self._log_to_linear(y)

        # 2) linear AP1 -> linear sRGB via W^{-1}
        if with_W:
            W_inv = self._get_inverse_W(lin_ap1.device, lin_ap1.dtype).view(3, 3, 1, 1)
            if channels_last:
                W_inv = W_inv.to(memory_format=torch.channels_last)
            lin_srgb = F.conv2d(lin_ap1, W_inv, bias=None, stride=1, padding=0)
        else:
            lin_srgb = lin_ap1

        out = lin_srgb

        return _from_chw(out, was_hwc)


