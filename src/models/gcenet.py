import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GRL, DropEdge, GCNLayer, SpatialAttentionPool, SEBlock


class GCE_Net(nn.Module):
    def __init__(
        self,
        A_norm,
        n_classes=3,
        n_domains=14,
        in_channels=5,
        input_time=235,
        num_electrodes=62,
        temp_filters=40,
        gcn_hidden=64,
        stage3_channels=64,
        se_reduction=4,
        supcon_dim=128,
        dropout=0.5,
        dropedge_p=0.2,
        use_gcn=True,
        use_dann1=True,
        use_dann2=True,
        use_supcon=True,
        use_spatial_pool=True,
    ):
        super().__init__()

        self.ablation = {
            "use_gcn": use_gcn,
            "use_dann1": use_dann1,
            "use_dann2": use_dann2,
            "use_supcon": use_supcon,
            "use_spatial_pool": use_spatial_pool,
        }

        self.register_buffer("A_norm", A_norm)
        self.n_domains = n_domains
        self.temp_filters = temp_filters
        self.gcn_hidden = gcn_hidden
        self.stage3_channels = stage3_channels
        self.num_electrodes = num_electrodes
        self.input_channels = in_channels
        self.input_time = input_time

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, temp_filters, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(temp_filters),
            nn.ELU(),
        )

        self.gcn = GCNLayer(temp_filters, gcn_hidden)
        self.residual_proj = nn.Linear(temp_filters, gcn_hidden)
        self.dropedge = DropEdge(p=dropedge_p)

        self.spatial_pool = SpatialAttentionPool(gcn_hidden)

        self.domain1_grl = GRL()
        self.domain1_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden, n_domains),
        )

        self.stage3_conv = nn.Sequential(
            nn.Conv2d(gcn_hidden, stage3_channels, kernel_size=(1, 16), groups=gcn_hidden, padding=(0, 8)),
            nn.Conv2d(stage3_channels, stage3_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(stage3_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        self.se = SEBlock(stage3_channels, reduction=se_reduction)

        self._compute_flatten()

        self.emotion_head = nn.Linear(self.flatten_size, n_classes)

        self.domain2_grl = GRL()
        self.domain2_head = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_domains),
        )

        self.supcon_head = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, supcon_dim),
        )

    def _compute_flatten(self):
        A_for_flatten = self.A_norm.detach().cpu()
        dummy = torch.zeros(1, self.input_channels, self.num_electrodes, self.input_time)
        x = self.stage1(dummy)
        B, _, V, actual_T = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * actual_T, V, self.temp_filters)
        x_gcn = self.gcn(x, A_for_flatten)
        x_res = self.residual_proj(x)
        x = x_gcn + x_res
        x_pooled, _ = self.spatial_pool(x)
        x = x_pooled.reshape(B, actual_T, self.gcn_hidden)
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.stage3_conv(x)
        x = self.se(x)
        self.flatten_size = x.reshape(1, -1).size(1)

    def forward(
        self,
        x,
        alpha=1.0,
        return_all=False,
        use_domain1=True,
        use_domain2=True,
        use_supcon=False,
        use_gcn=True,
        use_spatial_pool=True,
    ):
        B, C, V, _ = x.size()

        x = self.stage1(x)

        B, _, V, actual_T = x.size()

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * actual_T, V, self.temp_filters)

        if use_gcn:
            A_dropped = self.dropedge(self.A_norm, self.training)
            x_gcn = self.gcn(x, A_dropped)
            x_res = self.residual_proj(x)
            x = x_gcn + x_res
        else:
            x = self.residual_proj(x)

        if use_spatial_pool:
            x_pooled, attn_weights = self.spatial_pool(x)
            x = x_pooled.reshape(B, actual_T, self.gcn_hidden)
            x = x.permute(0, 2, 1).unsqueeze(2)
        else:
            x = x.reshape(B, actual_T, V, self.gcn_hidden).permute(0, 3, 2, 1)
            attn_weights = torch.zeros(B * actual_T, V, device=x.device)

        domain1_out = None
        if use_domain1:
            d1_feat = self.domain1_grl(x)
            domain1_out = self.domain1_head(d1_feat)

        x = self.stage3_conv(x)
        x = self.se(x)

        if not use_spatial_pool:
            x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))

        features = x.reshape(B, -1)

        emotion_out = self.emotion_head(features)

        domain2_out = None
        if use_domain2:
            d2_feat = self.domain2_grl(features)
            domain2_out = self.domain2_head(d2_feat)

        supcon_out = None
        if use_supcon:
            supcon_out = F.normalize(self.supcon_head(features), dim=1)

        if return_all:
            return emotion_out, domain1_out, domain2_out, supcon_out, features, attn_weights
        return emotion_out, domain1_out, domain2_out, supcon_out
