import torch
from rvc.v2_core.vendor.algorithm.modules import WaveNet
from rvc.v2_core.vendor.algorithm.commons import sequence_mask


class PosteriorEncoder(torch.nn.Module):
    def __init__(
        self, spec_channels, inter_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0
    ):
        super().__init__()
        self.conv_pre = torch.nn.Conv1d(spec_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, inter_channels * 2, 1)

    def forward(self, y, y_lengths, g=None):
        y_mask = sequence_mask(y_lengths, y.size(2)).unsqueeze(1).to(y.dtype)
        y = self.conv_pre(y) * y_mask
        y = self.enc(y, y_mask, g=g)
        stats = self.proj(y) * y_mask
        m, logs = torch.chunk(stats, 2, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * y_mask
        return z, m, logs, y_mask


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        hidden_dim: int,
        f0: bool = True,
    ):
        super().__init__()
        self.f0 = f0
        self.pre = torch.nn.Conv1d(hidden_dim + (2 if f0 else 0), hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate=1,
            n_layers=n_layers,
            p_dropout=p_dropout,
            gin_channels=0,
        )
        self.proj = torch.nn.Conv1d(hidden_channels, inter_channels * 2, 1)

    def forward(self, phone, pitch, phone_lengths):
        x_mask = sequence_mask(phone_lengths, phone.size(1)).unsqueeze(1).to(phone.dtype)
        if self.f0 and pitch is not None:
            phone = torch.cat([phone.transpose(1, 2), pitch.unsqueeze(1).float()], dim=1)
        else:
            phone = phone.transpose(1, 2)
        x = self.pre(phone) * x_mask
        x = self.enc(x, x_mask, g=None)
        stats = self.proj(x) * x_mask
        m_p, logs_p = torch.chunk(stats, 2, dim=1)
        return m_p, logs_p, x_mask


