import torch
import numpy as np
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

from rvc.v2_core.vendor.algorithm.residuals import LRELU_SLOPE, ResBlock
from rvc.v2_core.vendor.algorithm.commons import init_weights


class HiFiGANGenerator(torch.nn.Module):
    """
    HiFi-GAN Generator
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int = 0,
    ):
        super(HiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def __prepare_scriptable__(self):
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SineGenerator(torch.nn.Module):
    """
    Sine wave generator with optional harmonics and noise.
    """

    def __init__(
        self,
        sampling_rate: int,
        num_harmonics: int = 0,
        sine_amplitude: float = 0.1,
        noise_stddev: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super(SineGenerator, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = self.num_harmonics + 1

    def _compute_voiced_unvoiced(self, f0: torch.Tensor):
        uv_mask = (f0 > self.voiced_threshold).float()
        return uv_mask

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int):
        batch_size, length, _ = f0.shape
        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device
        )
        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += torch.nn.functional.pad(
            cumulative_phase, (0, 0, 1, 0), mode="constant"
        )
        phase_increments = phase_increments.reshape(batch_size, -1, 1)
        harmonic_scale = torch.arange(
            1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device
        ).reshape(1, 1, -1)
        phase_increments *= harmonic_scale
        random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
        random_phase[..., 0] = 0
        phase_increments += random_phase
        sine_waves = torch.sin(2 * np.pi * phase_increments)
        return sine_waves

    def forward(self, f0: torch.Tensor, upsampling_factor: int):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
            voiced_mask = self._compute_voiced_unvoiced(f0)
            voiced_mask = torch.nn.functional.interpolate(
                voiced_mask.transpose(2, 1),
                scale_factor=float(upsampling_factor),
                mode="nearest",
            ).transpose(2, 1)
            noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (
                self.sine_amplitude / 3
            )
            noise = noise_amplitude * torch.randn_like(sine_waves)
            sine_waveforms = sine_waves * voiced_mask + noise
        return sine_waveforms, voiced_mask, noise


