from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_index_select(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # features: [B, C, N], indices: [B, M, K] -> [B, C, M, K]
    b, c, _ = features.shape
    _, m, k = indices.shape
    idx = indices.unsqueeze(1).expand(b, c, m, k)
    return torch.gather(features.unsqueeze(2).expand(b, c, m, features.shape[-1]), 3, idx)


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    # xyz: [B, N, 3] -> idx: [B, N, k]
    dist = torch.cdist(xyz, xyz)
    idx = dist.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]
    return idx


class LocalFeatureAggregation(nn.Module):
    """LFA with attentive pooling instead of max-pooling.

    Attention pooling lets the network learn *which* neighbours matter most,
    which is especially beneficial for thin, sparse structures (power lines).
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        hidden = out_channels // 2

        self.pre_mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.edge_mlp = nn.Sequential(
            nn.Conv2d(hidden * 2 + 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Attentive pooling: learn a scalar weight per neighbour
        self.att_score = nn.Conv2d(out_channels, 1, kernel_size=1, bias=True)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # xyz: [B, N, 3], features: [B, C, N]
        idx = knn_indices(xyz, self.k)

        f = self.pre_mlp(features)
        neighbor_f = batched_index_select(f, idx)

        center_f = f.unsqueeze(-1).expand_as(neighbor_f)

        xyz_t = xyz.transpose(1, 2)
        neighbor_xyz = batched_index_select(xyz_t, idx)
        center_xyz = xyz_t.unsqueeze(-1).expand_as(neighbor_xyz)

        rel_xyz = neighbor_xyz - center_xyz
        dist = torch.norm(rel_xyz, dim=1, keepdim=True)

        edge_feat = torch.cat([center_f, neighbor_f - center_f, rel_xyz, dist], dim=1)
        edge_out = self.edge_mlp(edge_feat)          # [B, C_out, N, k]

        # Attentive pooling: softmax over k neighbours, then weighted sum
        att = torch.softmax(self.att_score(edge_out), dim=-1)  # [B, 1, N, k]
        out = (edge_out * att).sum(dim=-1)           # [B, C_out, N]

        shortcut = self.shortcut(features)
        return F.leaky_relu(out + shortcut, negative_slope=0.2, inplace=True)


def random_subsample(xyz: torch.Tensor, features: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    b, n, _ = xyz.shape
    m = max(32, int(n * ratio))

    rand = torch.rand(b, n, device=xyz.device)
    idx = rand.topk(k=m, dim=1).indices

    idx_xyz = idx.unsqueeze(-1).expand(b, m, 3)
    sub_xyz = torch.gather(xyz, 1, idx_xyz)

    idx_feat = idx.unsqueeze(1).expand(b, features.shape[1], m)
    sub_feat = torch.gather(features, 2, idx_feat)

    return sub_xyz, sub_feat


def nearest_interpolate(target_xyz: torch.Tensor, source_xyz: torch.Tensor, source_feat: torch.Tensor) -> torch.Tensor:
    # target_xyz: [B, Nt, 3], source_xyz: [B, Ns, 3], source_feat: [B, C, Ns]
    dist = torch.cdist(target_xyz, source_xyz)
    nn_idx = dist.argmin(dim=-1)

    idx = nn_idx.unsqueeze(1).expand(target_xyz.shape[0], source_feat.shape[1], target_xyz.shape[1])
    return torch.gather(source_feat, 2, idx)


class RandLANet(nn.Module):
    def __init__(self, num_classes: int = 2, k: int = 16, d_model: list = None) -> None:
        super().__init__()
        if d_model is None:
            d_model = [32, 64, 128, 256]  # Match checkpoint dimensions
        # Encoder
        self.enc1 = LocalFeatureAggregation(3, d_model[0], k=k)
        self.enc2 = LocalFeatureAggregation(d_model[0], d_model[1], k=k)
        self.enc3 = LocalFeatureAggregation(d_model[1], d_model[2], k=k)
        self.enc4 = LocalFeatureAggregation(d_model[2], d_model[3], k=k)

        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv1d(d_model[3] + d_model[2], 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(128 + d_model[1], 96, kernel_size=1, bias=False),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(96 + d_model[0], 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Classification head — two-stage MLP with dropout at both stages
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv1d(64, num_classes, kernel_size=1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: [B, N, 3]
        x0 = xyz.transpose(1, 2)

        f1 = self.enc1(xyz, x0)

        xyz2, f2_in = random_subsample(xyz, f1, ratio=0.25)
        f2 = self.enc2(xyz2, f2_in)

        xyz3, f3_in = random_subsample(xyz2, f2, ratio=0.25)
        f3 = self.enc3(xyz3, f3_in)

        xyz4, f4_in = random_subsample(xyz3, f3, ratio=0.25)
        f4 = self.enc4(xyz4, f4_in)

        up3 = nearest_interpolate(xyz3, xyz4, f4)
        d3 = self.dec3(torch.cat([up3, f3], dim=1))

        up2 = nearest_interpolate(xyz2, xyz3, d3)
        d2 = self.dec2(torch.cat([up2, f2], dim=1))

        up1 = nearest_interpolate(xyz, xyz2, d2)
        d1 = self.dec1(torch.cat([up1, f1], dim=1))

        logits = self.head(d1)
        return logits
