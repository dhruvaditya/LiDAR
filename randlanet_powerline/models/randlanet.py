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
        out = self.edge_mlp(edge_feat).max(dim=-1).values

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
    def __init__(self, num_classes: int = 2, k: int = 16) -> None:
        super().__init__()
        self.enc1 = LocalFeatureAggregation(3, 32, k=k)
        self.enc2 = LocalFeatureAggregation(32, 64, k=k)
        self.enc3 = LocalFeatureAggregation(64, 128, k=k)

        self.dec2 = nn.Sequential(
            nn.Conv1d(128 + 64, 96, kernel_size=1, bias=False),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(96 + 32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

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

        up2 = nearest_interpolate(xyz2, xyz3, f3)
        d2 = self.dec2(torch.cat([up2, f2], dim=1))

        up1 = nearest_interpolate(xyz, xyz2, d2)
        d1 = self.dec1(torch.cat([up1, f1], dim=1))

        logits = self.head(d1)
        return logits
