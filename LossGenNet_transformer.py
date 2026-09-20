import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


LGN_WEIGHT_NAMES = (
    "avoidance",
    "exploration",
    "turn",
    "progress",
    "smoothness",
    "energy",
)
LGN_SAFETY_WEIGHT_COUNT = 3


class LossGenNet(nn.Module):
    """Causal Transformer LGN restored from the requested experiment snapshot."""

    def __init__(
        self,
        state_dim,
        geom_dim=19,
        progress_dim=8,
        hidden_dim=128,
        nhead=4,
        num_layers=2,
        max_seq_len=64,
        output_temperature=1.0,
        weight_floor=0.01,
        preference_weight_floor=0.01,
        weight_ceiling=5.0,
        activation_checkpoint=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.activation_checkpoint = bool(activation_checkpoint)
        self.output_temperature = float(output_temperature)
        self.weight_floor = float(weight_floor)
        self.preference_weight_floor = float(preference_weight_floor)
        self.weight_ceiling = float(weight_ceiling)
        if self.output_temperature <= 0.0:
            raise ValueError("output_temperature must be positive")
        if self.weight_floor < 0.0:
            raise ValueError("weight_floor must be non-negative")
        if self.preference_weight_floor < 0.0:
            raise ValueError("preference_weight_floor must be non-negative")
        if self.weight_ceiling <= self.weight_floor:
            raise ValueError("weight_ceiling must be greater than weight_floor")
        if self.weight_ceiling <= self.preference_weight_floor:
            raise ValueError("weight_ceiling must be greater than preference_weight_floor")

        self.visual_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> [32, 12, 16]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # -> [32, 6, 8]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [64, 6, 8]
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # -> [64, 3, 4]
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.visual_proj = nn.Linear(64 * 3 * 4, hidden_dim)

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
        )
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.Tanh(),
        )
        self.progress_proj = nn.Sequential(
            nn.Linear(progress_dim, hidden_dim),
            nn.Tanh(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

        # All six weights retain a small positive floor by default.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, len(LGN_WEIGHT_NAMES)),
        )
        with torch.no_grad():
            # Start newly introduced preferences close to off. This also gives
            # checkpoint expansion a safe initialization for unmapped rows.
            self.head[2].weight[LGN_SAFETY_WEIGHT_COUNT:].mul_(0.01)
            self.head[2].bias[LGN_SAFETY_WEIGHT_COUNT:].fill_(-4.0)

    def forward(self, depth_feat, state, geom_feat, progress_feat, hx=None):
        """
        Args:
            depth_feat: [B, 1, 12, 16] depth feature map.
            state: [B, state_dim] physical state.
            geom_feat: [B, geom_dim] geometry/risk features.
            progress_feat: [B, progress_dim] recent progress features.
            hx: [B, T_mem, hidden_dim] raw-token memory, or None.
        Returns:
            outputs: [B, 6] Avoidance/Exploration/Turn/Progress/
                Smoothness/Energy loss weights.
            memory: [B, T_mem, hidden_dim] updated raw-token memory.
        """
        v_emb = self.visual_proj(self.visual_net(depth_feat))
        s_emb = self.state_proj(state)
        g_emb = self.geom_proj(geom_feat)
        p_emb = self.progress_proj(progress_feat)

        fused = torch.cat([v_emb, s_emb, g_emb, p_emb], dim=-1)
        current_token = self.pre_norm(self.fusion_proj(fused))

        if hx is None:
            seq = current_token.unsqueeze(1)
        else:
            seq = torch.cat([hx, current_token.unsqueeze(1)], dim=1)
            if seq.size(1) > self.max_seq_len:
                seq = seq[:, -self.max_seq_len:]

        seq_len = seq.size(1)
        x_in = seq + self.pos_emb[:, :seq_len]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x_in.device, dtype=torch.bool),
            diagonal=1,
        )
        if (
            self.activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        ):
            x_out = checkpoint(
                lambda xin: self.transformer(xin, mask=causal_mask),
                x_in,
                use_reentrant=False,
            )
        else:
            x_out = self.transformer(x_in, mask=causal_mask)
        last_token = self.out_norm(x_out[:, -1])
        raw = self.head(last_token)

        positive = F.softplus(raw / self.output_temperature) * self.output_temperature
        safety_span = self.weight_ceiling - self.weight_floor
        safety_outputs = self.weight_floor + safety_span * torch.tanh(
            positive[:, :LGN_SAFETY_WEIGHT_COUNT] / safety_span
        )
        preference_span = self.weight_ceiling - self.preference_weight_floor
        preference_outputs = self.preference_weight_floor + preference_span * torch.tanh(
            positive[:, LGN_SAFETY_WEIGHT_COUNT:] / preference_span
        )
        outputs = torch.cat([safety_outputs, preference_outputs], dim=-1)

        # Preserve raw fused tokens, matching the requested snapshot model.
        return outputs, seq
