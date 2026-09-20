"""Spatially enhanced recurrent Worker for depth-based flight control."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest useful GroupNorm group count that divides channels."""
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class SpatiallyEnhancedRecurrentUnit(nn.Module):
    """ConvGRU-style cell with local/dilated spatial candidate fusion.

    Unlike a vector recurrent cell, both the memory and all recurrent gates keep
    the image grid.  The dilated candidate branch gives each update a wider
    obstacle context without discarding left/right/up/down location.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        pair_channels = self.channels * 2
        groups = _group_count(self.channels)

        self.gates = nn.Conv2d(pair_channels, pair_channels, kernel_size=3, padding=1)
        self.local_candidate = nn.Conv2d(
            pair_channels, self.channels, kernel_size=3, padding=1
        )
        self.context_candidate = nn.Conv2d(
            pair_channels,
            self.channels,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.spatial_gate = nn.Conv2d(
            self.channels * 3, self.channels, kernel_size=1
        )
        self.candidate_norm = nn.GroupNorm(groups, self.channels)

        # Begin with a conservative update rate so early noisy frames do not
        # immediately overwrite the complete spatial memory.
        with torch.no_grad():
            self.gates.bias[self.channels :].fill_(-1.0)

    def forward(self, current: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        reset_gate, update_gate = torch.sigmoid(
            self.gates(torch.cat([current, memory], dim=1))
        ).chunk(2, dim=1)

        candidate_input = torch.cat([current, reset_gate * memory], dim=1)
        local = self.local_candidate(candidate_input)
        context = self.context_candidate(candidate_input)
        spatial_weight = torch.sigmoid(
            self.spatial_gate(torch.cat([local, context, memory], dim=1))
        )
        candidate = torch.tanh(
            self.candidate_norm(local + spatial_weight * context)
        )
        return (1.0 - update_gate) * memory + update_gate * candidate


class WorkNet(nn.Module):
    """Worker using fixed-size spatial recurrent memory.

    The public interface intentionally matches ``WorkNet_transformer.WorkNet``
    so the main rollout and differentiable meta-rollout can switch backbones
    without changing their control or loss logic.
    """

    def __init__(
        self,
        dim_obs: int = 9,
        dim_action: int = 4,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 2,
        max_seq_len: int = 64,
        activation_checkpoint: bool = False,
        hidden_channels: int = 96,
    ) -> None:
        super().__init__()
        del nhead, num_layers  # Accepted for drop-in constructor compatibility.
        if hidden_channels < 8:
            raise ValueError("hidden_channels must be at least 8")

        self.hidden_channels = int(hidden_channels)
        self.max_seq_len = int(max_seq_len)  # Retained in checkpoints/configs.
        self.activation_checkpoint = bool(activation_checkpoint)
        encoder_channels = max(16, self.hidden_channels // 2)
        groups = _group_count(self.hidden_channels)

        # Input is [B, 1, 12, 16]. Preserve a [B, C, 6, 8] grid instead of
        # flattening it into the single token used by the Transformer Worker.
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, encoder_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(
                encoder_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.LeakyReLU(0.05),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(dim_obs, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.SiLU(),
        )
        self.state_film = nn.Linear(self.hidden_channels, self.hidden_channels * 2)
        nn.init.zeros_(self.state_film.weight)
        nn.init.zeros_(self.state_film.bias)

        self.recurrent = SpatiallyEnhancedRecurrentUnit(self.hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_channels * 2),
            nn.Linear(self.hidden_channels * 2, d_model),
            nn.LeakyReLU(0.05),
            nn.Linear(d_model, dim_action, bias=False),
        )
        self.action_head[-1].weight.data.mul_(0.01)

    def reset(self) -> None:
        # Memory is passed explicitly by the rollout; no module-global state.
        return None

    def forward(self, x: torch.Tensor, v: torch.Tensor, hx=None):
        spatial = self.depth_encoder(x)
        state = self.state_encoder(v)
        gamma, beta = self.state_film(state).chunk(2, dim=-1)
        gamma = 0.5 * torch.tanh(gamma)
        spatial = spatial * (1.0 + gamma[:, :, None, None])
        spatial = spatial + beta[:, :, None, None]

        if hx is None:
            hx = torch.zeros_like(spatial)
        elif hx.shape != spatial.shape:
            raise ValueError(
                f"SRU memory shape {tuple(hx.shape)} does not match "
                f"current feature shape {tuple(spatial.shape)}"
            )

        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            memory = checkpoint(self.recurrent, spatial, hx, use_reentrant=False)
        else:
            memory = self.recurrent(spatial, hx)

        pooled_memory = self.pool(memory).flatten(1)
        action = self.action_head(torch.cat([pooled_memory, state], dim=-1))
        return action, None, memory


if __name__ == "__main__":
    WorkNet()
