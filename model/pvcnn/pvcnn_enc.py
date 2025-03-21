"""
use PVCNN to get a latent vector: as the encoder of an autoencoder

Encoder layer 0, output feature shape: torch.Size([32, 64, 1024])
Encoder layer 1, output feature shape: torch.Size([32, 128, 256])
Encoder layer 2, output feature shape: torch.Size([32, 256, 64])
Encoder layer 3, output feature shape: torch.Size([32, 512, 16])
Encoder layer 4, output feature shape: torch.Size([32, 1024, 4])
Encoder layer 5, output feature shape: torch.Size([32, 1024, 1])
"""
import torch
import torch.nn as nn

from .pvcnn import PVCNN2, PVCNN2Base
from model.pvcnn.pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules


class PVCNNEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        # conv_configs, sa_configs
        # conv_configs: (out_ch, num_blocks, voxel_reso), sa_configs: (num_centers, radius, num_neighbors, out_channels)
        sa_blocks = [
            ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
            ((64, 3, 16), (256, 0.2, 32, (64, 128))),
            ((128, 3, 8), (64, 0.4, 32, (128, 256))),
            # (None, (16, 0.8, 32, (256, 256, 512))),
            ((256, 3, 4), (16, 0.8, 32, (256, 256, 512))), # output: 512x16
            (None, (4, 0.8, 8, (512, 1024))), # output: 1024x4
            (None, (1, 0.8, 4, (1024, latent_dim))), # output: 1024x1
        ]
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks_config=sa_blocks,
            extra_feature_channels=0,
            with_se=True,
            embed_dim=3, # no timestamp embedding, only the point coordinates
            use_att=True,
            # dropout=dropout,
            # width_multiplier=width_multiplier,
            # voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """
        inputs: (B, 3, N)
        outputs: (B, D)
        """
        bs = inputs.shape[0]
        # feat = self.sa_layers(inputs)
        # return feat.reshape(bs, -1)

        features = inputs # (B, 3, N)
        coords = inputs[:, :3, :].contiguous()
        t_emb = coords.clone()

        coords_list = []
        in_features_list = []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, t_emb = sa_blocks((features, coords, t_emb))
            else:
                features, coords, t_emb = sa_blocks((torch.cat([features, t_emb], dim=1), coords, t_emb))
            # print(f"Encoder layer {i}, output feature shape: {features.shape}")
        return features.reshape(bs, -1)

class ResBlock(nn.Module):
    "one residual block"
    def __init__(self, in_ch, hidden_dim):
        """
        3 layer residual block
        (*, in_ch) -> (*, hidden) -> (*, hidden) -> (*, in_ch)
        :param in_ch:
        :param hidden_dim:
        """
        super().__init__()
        modules = [
            nn.Linear(in_ch, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, in_ch),
            nn.LeakyReLU()
        ]
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.layers(x)


class PVCNNAutoEncoder(nn.Module):
    def __init__(self, latent_dim=1024, num_dec_blocks=6, block_dims=[512, 256], num_points=1500,
                 bottleneck_dim=-1):
        """

        :param latent_dim:
        :param num_dec_blocks:
        :param block_dims:
        :param num_points:
        :param bottleneck_dim: the input dimension to the last MLP layer, a lower bottleneck
        """
        super().__init__()
        self.encoder = PVCNNEncoder(latent_dim)
        modules = [nn.Linear(latent_dim, block_dims[0]), nn.LeakyReLU()]
        modules.extend([ResBlock(*block_dims) for _ in range(num_dec_blocks)])

        # output layers
        if bottleneck_dim <=0:
            modules.append(nn.Linear(block_dims[0], num_points*3))
        else:
            print(f"Using bottleneck dimension={bottleneck_dim}")
            modules.append(nn.Linear(block_dims[0], bottleneck_dim))
            modules.append(nn.Linear(bottleneck_dim, num_points*3))
        self.decoder = nn.Sequential(*modules)
        self.num_points = num_points
        self.latents = None

    def forward(self, x, ret_latent=False):
        """
        x: (B, N, 3)
        output: (B, N_out, 3)
        """
        B, N = x.shape[:2]
        latent = self.encoder(x.permute(0, 2, 1), None) # (B, D)
        output = self.decoder(latent)
        self.latents = latent # cache

        if ret_latent:
            return output.reshape(B, self.num_points, 3), latent

        return output.reshape(B, self.num_points, 3)

    def encode(self, x):
        "encode input points as latent code, return (B, D)"
        latent = self.encoder(x.permute(0, 2, 1), None)
        return latent

    def decode(self, latent):
        """
        decode latent codes to points
        latent: (B, D)
        return: (B, N, 3)
        """
        B = latent.shape[0]
        output = self.decoder(latent)
        self.latents = latent

        return output.reshape(B, self.num_points, 3)