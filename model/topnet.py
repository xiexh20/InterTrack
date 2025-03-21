"""
Temporal based object pose prediction network (TOPNet)
"""
import torch
import numpy as np
import torch.nn as nn
from .posi_embed import PositionEmbeddingSine_1D

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi # Bx1, 1xD -> BxD
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # Bx2D


class RotationModel(nn.Module):
    def __init__(self, input_dim, embed_dim, d_feedforward=256,
                 pose_dim=6, # input pose dimension
                 out_dim=6, # output pose prediction dimension
                 pose_feat_dim=128,
                 num_layers=3, nhead=4,
                 block_dims=[128, 64],
                 norm='none',
                 add_src_key_mask=-1.):
        super().__init__()
        # TODO: hyperparameters that we can tune: feedforward dimension, src_key_mask use or not
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead,
                                                    dim_feedforward=d_feedforward,
                                                   batch_first=True,
                                                   activation='gelu') # input (B, L, E)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.bnorm = nn.BatchNorm1d()
        self.decoder = self.init_decoder(embed_dim, out_dim)
        # t encoder, from: https://github.com/Jiyao06/GenPose/blob/main/networks/gf_algorithms/scorenet.py#L55
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=64),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )
        self.pose_encoder = self.init_pose_encoder(pose_dim, pose_feat_dim)
        self.posi_embed = PositionEmbeddingSine_1D(32)

        # Normalization over features
        img_feat_dim = 768
        self.gnorm_img = nn.GroupNorm(img_feat_dim//4, img_feat_dim)  # group normalization
        self.gnorm_pose = nn.GroupNorm(pose_feat_dim//4, pose_feat_dim)  # group normalization
        self.norm = norm
        self.bnorm_img = nn.BatchNorm1d(img_feat_dim)
        self.bnorm_pose = nn.BatchNorm1d(pose_feat_dim)
        self.bnorm_emb = nn.BatchNorm1d(embed_dim)
        self.layer_norm = nn.LayerNorm(img_feat_dim)
        self.add_src_key_mask = add_src_key_mask # the threshold to not attend to images where object is heavily occluded
        print("Normalization for rotation prediction model:", self.norm)

        if self.norm == 'layer':
            #add layer normalization after all activations
            self.linear_embedding = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim))
            self.t_encoder = nn.Sequential(
                GaussianFourierProjection(embed_dim=64),
                nn.Linear(64, 64),
                nn.LayerNorm(64),
            )
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_dim, 128),  # pose dimension is 6
                nn.LeakyReLU(),
                nn.Linear(128, pose_feat_dim),
                nn.LayerNorm(pose_feat_dim),
            )

    def init_decoder(self, embed_dim, out_dim):
        blocks = [
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_dim)
        ]
        return nn.Sequential(*blocks)  # *, embed_dim->128->64->6

    def init_pose_encoder(self, pose_dim, pose_feat_dim):
        return nn.Sequential(
            nn.Linear(pose_dim, 128),  # pose dimension is 6
            nn.LeakyReLU(),
            nn.Linear(128, pose_feat_dim),
            nn.LeakyReLU(),
        )

    def forward(self, pose6d, img_feats, t):
        """

        Parameters
        ----------
        pose6d : (B, T, 6), poses at step t
        img_feats : (B, T, D), global image features
        t : (B,) timesteps for each example

        Returns (B, T, out_dim)
        -------

        feature ranges before doing any normalization: (reg16 after training)
        Ranges: pose -0.0043-0.5029, img -72.5601-76.6325, positional encoding -0.9968-1.0000, time -0.0130-1.0221
        Ranges: pose -0.0043-0.5029, img -59.3237-72.9365, positional encoding -0.9968-1.0000, time -0.0130-1.0221

        Frame 0 ranges: pose -0.0043-0.5029, img -68.7879-72.4790, positional encoding 0.0000-1.0000, time -0.0130-1.0221,img feature norm: -1.8809-1.8580
        Frame 1 ranges: pose -0.0043-0.5029, img -68.8912-72.5654, positional encoding 0.0000-1.0000, time -0.0130-1.0221,img feature norm: -1.8837-1.8602
        Frame 2 ranges: pose -0.0043-0.5029, img -67.3467-71.0159, positional encoding 0.0000-1.0000, time -0.0130-1.0221,img feature norm: -1.8414-1.8196
        Frame 3 ranges: pose -0.0043-0.5029, img -65.2828-68.7007, positional encoding 0.0000-1.0000, time -0.0130-1.0221,img feature norm: -1.7848-1.7966
        Frame 4 ranges: pose -0.0043-0.5029, img -64.1957-67.6431, positional encoding -0.1045-1.0000, time -0.0130-1.0221,img feature norm: -1.7551-1.7979
        Frame 5 ranges: pose -0.0043-0.5029, img -64.7283-68.1983, positional encoding -0.5000-1.0000, time -0.0130-1.0221,img feature norm: -1.7697-1.8109
        Frame 6 ranges: pose -0.0043-0.5029, img -66.0956-69.7629, positional encoding -0.8090-1.0000, time -0.0130-1.0221,img feature norm: -1.8071-1.8197
        Frame 7 ranges: pose -0.0043-0.5029, img -67.1150-70.9413, positional encoding -0.9781-1.0000, time -0.0130-1.0221,img feature norm: -1.8350-1.8177
        Frame 8 ranges: pose -0.0043-0.5029, img -67.9375-71.7766, positional encoding -0.9781-1.0000, time -0.0130-1.0221,img feature norm: -1.8576-1.8396
        Frame 9 ranges: pose -0.0043-0.5029, img -68.4981-72.6904, positional encoding -0.8090-1.0000, time -0.0130-1.0221,img feature norm: -1.8729-1.8635
        Frame 10 ranges: pose -0.0043-0.5029, img -68.7299-73.1467, positional encoding -0.8660-1.0000, time -0.0130-1.0221,img feature norm: -1.8793-1.8755
        Frame 11 ranges: pose -0.0043-0.5029, img -69.6626-73.9974, positional encoding -0.9945-1.0000, time -0.0130-1.0221,img feature norm: -1.9048-1.8977
        Frame 12 ranges: pose -0.0043-0.5029, img -68.6573-72.7827, positional encoding -0.9511-1.0000, time -0.0130-1.0221,img feature norm: -1.8789-1.8659
        Frame 13 ranges: pose -0.0043-0.5029, img -70.6611-74.9230, positional encoding -0.9968-1.0000, time -0.0130-1.0221,img feature norm: -1.9389-1.9220
        Frame 14 ranges: pose -0.0043-0.5029, img -72.5601-76.6325, positional encoding -0.9878-1.0000, time -0.0130-1.0221,img feature norm: -1.9966-1.9668
        Frame 15 ranges: pose -0.0043-0.5029, img -71.8525-75.4763, positional encoding -0.9243-1.0000, time -0.0130-1.0221,img feature norm: -1.9696-1.9365

        # after group normalization:
        Frame 0 ranges: pose -1.5790-1.7319, img -2.4675-1.9304, positional encoding 0.0000-1.0000, time -0.0130-1.0221,
        Frame 1 ranges: pose -1.5790-1.7319, img -2.5795-1.9728, positional encoding 0.0000-1.0000, time -0.0130-1.0221,
        Frame 2 ranges: pose -1.5790-1.7319, img -2.1974-1.8752, positional encoding 0.0000-1.0000, time -0.0130-1.0221,
        Frame 3 ranges: pose -1.5790-1.7319, img -2.2520-1.8928, positional encoding 0.0000-1.0000, time -0.0130-1.0221,
        Frame 4 ranges: pose -1.5790-1.7319, img -1.7914-1.7802, positional encoding -0.1045-1.0000, time -0.0130-1.0221,
        Frame 5 ranges: pose -1.5790-1.7319, img -1.8729-1.7652, positional encoding -0.5000-1.0000, time -0.0130-1.0221,
        Frame 6 ranges: pose -1.5790-1.7319, img -1.7911-1.7474, positional encoding -0.8090-1.0000, time -0.0130-1.0221,
        Frame 7 ranges: pose -1.5790-1.7319, img -1.8795-1.7242, positional encoding -0.9781-1.0000, time -0.0130-1.0221,
        Frame 8 ranges: pose -1.5790-1.7319, img -1.7882-1.7637, positional encoding -0.9781-1.0000, time -0.0130-1.0221,
        Frame 9 ranges: pose -1.5790-1.7319, img -1.7703-1.7626, positional encoding -0.8090-1.0000, time -0.0130-1.0221,
        Frame 10 ranges: pose -1.5790-1.7319, img -1.8914-1.7488, positional encoding -0.8660-1.0000, time -0.0130-1.0221,
        Frame 11 ranges: pose -1.5790-1.7319, img -1.8422-1.9347, positional encoding -0.9945-1.0000, time -0.0130-1.0221,
        Frame 12 ranges: pose -1.5790-1.7319, img -1.7554-1.8659, positional encoding -0.9511-1.0000, time -0.0130-1.0221,
        Frame 13 ranges: pose -1.5790-1.7319, img -1.7906-1.8113, positional encoding -0.9968-1.0000, time -0.0130-1.0221,
        Frame 14 ranges: pose -1.5790-1.7319, img -2.0910-1.8476, positional encoding -0.9878-1.0000, time -0.0130-1.0221,
        Frame 15 ranges: pose -1.5790-1.7319, img -1.8403-2.0437, positional encoding -0.9243-1.0000, time -0.0130-1.0221,


        reg16 overfitting:
        Frame 0 ranges: pose -0.0032-0.2823, img -25.6564-24.2910, positional encoding 0.0000-1.0000, time -0.0116-0.9891
        Frame 1 ranges: pose -0.0032-0.2823, img -25.3684-24.6178, positional encoding 0.0000-1.0000, time -0.0116-0.9891
        Frame 2 ranges: pose -0.0032-0.2823, img -24.2236-25.4861, positional encoding 0.0000-1.0000, time -0.0116-0.9891
        Frame 3 ranges: pose -0.0032-0.2823, img -24.3717-24.8913, positional encoding 0.0000-1.0000, time -0.0116-0.9891
        Frame 4 ranges: pose -0.0032-0.2823, img -24.6446-24.2929, positional encoding -0.1045-1.0000, time -0.0116-0.9891
        Frame 5 ranges: pose -0.0032-0.2823, img -25.1225-24.1236, positional encoding -0.5000-1.0000, time -0.0116-0.9891
        Frame 6 ranges: pose -0.0032-0.2823, img -25.5420-24.3765, positional encoding -0.8090-1.0000, time -0.0116-0.9891
        Frame 7 ranges: pose -0.0032-0.2823, img -25.4549-24.6451, positional encoding -0.9781-1.0000, time -0.0116-0.9891
        Frame 8 ranges: pose -0.0032-0.2823, img -25.0347-24.4826, positional encoding -0.9781-1.0000, time -0.0116-0.9891
        Frame 9 ranges: pose -0.0032-0.2823, img -24.6465-24.0997, positional encoding -0.8090-1.0000, time -0.0116-0.9891
        Frame 10 ranges: pose -0.0032-0.2823, img -24.3088-24.6011, positional encoding -0.8660-1.0000, time -0.0116-0.9891
        Frame 11 ranges: pose -0.0032-0.2823, img -24.1014-24.6915, positional encoding -0.9945-1.0000, time -0.0116-0.9891
        Frame 12 ranges: pose -0.0032-0.2823, img -23.6650-24.0640, positional encoding -0.9511-1.0000, time -0.0116-0.9891
        Frame 13 ranges: pose -0.0032-0.2823, img -23.3289-23.3447, positional encoding -0.9968-1.0000, time -0.0116-0.9891
        Frame 14 ranges: pose -0.0032-0.2823, img -21.8198-21.7414, positional encoding -0.9878-1.0000, time -0.0116-0.9891
        Frame 15 ranges: pose -0.0032-0.2823, img -20.9140-21.0307, positional encoding -0.9243-1.0000, time -0.0116-0.9891

        After adding SMPL conditioning:
        Frame 0 ranges: pose -1.3368-1.5289, img -2.7816-3.2075, positional encoding 0.0000-1.0000, time -0.0089-2.0243,embed -2.7159-1.8932714462280273
        Frame 1 ranges: pose -1.4246-1.5869, img -3.0080-3.2281, positional encoding 0.0000-1.0000, time -0.0089-2.0243,embed -2.9849-1.9327621459960938
        Frame 2 ranges: pose -1.4235-1.5995, img -2.9139-3.2053, positional encoding 0.0000-1.0000, time -0.0089-2.0243,embed -2.7501-1.96180260181427
        Frame 3 ranges: pose -1.1847-1.2704, img -3.2321-3.1985, positional encoding 0.0000-1.0000, time -0.0089-2.0243,embed -2.6608-2.2976367473602295
        Frame 4 ranges: pose -1.1685-1.1631, img -3.1097-3.1614, positional encoding -0.1045-1.0000, time -0.0089-2.0243,embed -2.5510-2.2012178897857666
        Frame 5 ranges: pose -1.3273-1.1364, img -3.3003-3.1327, positional encoding -0.5000-1.0000, time -0.0089-2.0243,embed -2.8237-1.8925492763519287
        Frame 6 ranges: pose -1.5302-1.3614, img -2.8359-3.1026, positional encoding -0.8090-1.0000, time -0.0089-2.0243,embed -2.8287-1.9130643606185913
        Frame 7 ranges: pose -1.4189-1.4884, img -2.6755-3.0378, positional encoding -0.9781-1.0000, time -0.0089-2.0243,embed -2.9479-1.9851305484771729
        Frame 8 ranges: pose -1.3812-1.2848, img -2.7712-2.9436, positional encoding -0.9781-1.0000, time -0.0089-2.0243,embed -2.7832-2.0617947578430176
        Frame 9 ranges: pose -1.2563-1.2537, img -2.7910-2.8259, positional encoding -0.8090-1.0000, time -0.0089-2.0243,embed -2.7502-2.263573408126831
        Frame 10 ranges: pose -1.5173-1.3410, img -2.5535-2.8046, positional encoding -0.8660-1.0000, time -0.0089-2.0243,embed -2.4742-2.1310250759124756
        Frame 11 ranges: pose -1.4725-1.2878, img -2.6345-2.9480, positional encoding -0.9945-1.0000, time -0.0089-2.0243,embed -2.5733-2.153665065765381
        Frame 12 ranges: pose -1.5657-1.3278, img -2.6357-3.1927, positional encoding -0.9511-1.0000, time -0.0089-2.0243,embed -2.6808-2.049337863922119
        Frame 13 ranges: pose -1.4875-1.4222, img -2.6812-3.2214, positional encoding -0.9968-1.0000, time -0.0089-2.0243,embed -2.5902-2.1664135456085205
        Frame 14 ranges: pose -1.3724-1.5027, img -2.6392-3.3419, positional encoding -0.9878-1.0000, time -0.0089-2.0243,embed -2.7637-2.2857894897460938
        Frame 15 ranges: pose -1.4800-1.4408, img -2.7242-3.2182, positional encoding -0.9243-1.0000, time -0.0089-2.0243,embed -2.3818-2.4355826377868652

        """
        B, T = pose6d.shape[:2]
        pose_feat = self.pose_encoder(pose6d)
        t_feat = self.t_encoder(t)[:, None].repeat(1, T, 1) # (B,) -> (B, D) -> (B, T, D)
        posi = self.posi_embed(B, T).to(pose6d.device) # (B, T) -> (B, T, D)
        if self.norm == 'group':
            # Apply group normalization
            # print("Applying group normalization layer to image features ")
            vis = img_feats[:, :, -1:]
            img_feat_norm = self.gnorm_img(img_feats[:, :, :-1].permute(0, 2, 1)).permute(0, 2, 1)
            img_feats = torch.cat([img_feat_norm, vis], -1) # (B, T, D)
            pose_feat = self.gnorm_pose(pose_feat.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.norm in ['batch', 'batch-emb']:
            # Apply batch normalization
            vis = img_feats[:, :, -1:]
            img_feat_norm = self.bnorm_img(img_feats[:, :, :-1].permute(0, 2, 1)).permute(0, 2, 1)
            img_feats = torch.cat([img_feat_norm, vis], -1)  # (B, T, D)
            pose_feat = self.bnorm_pose(pose_feat.permute(0, 2, 1)).permute(0, 2, 1) # bnorm also on the embeded human pose
        elif self.norm == 'layer':
            # apply layer norm on image features
            img_feat_norm = self.layer_norm(img_feats[:, :, :-1])
            img_feats = torch.cat([img_feat_norm, img_feats[:, :, -1:]], -1)
        # visualize pose feature
        # import glob
        # from tool.plt_utils import PltVisualizer
        # name = '15fps-pose-feat'
        # fcount = len(glob.glob(f'/BS/xxie-2/work/pc2-diff/experiments/debug/images/{name}_*.png'))
        # for ii, feat in enumerate(pose_feat):
        #     feat_np = feat.cpu().numpy().T # (T, D)->(D, T)
        #     outfile = f'/BS/xxie-2/work/pc2-diff/experiments/debug/images/{name}_{fcount}_{ii:02d}.png'
        #     PltVisualizer.plot_and_save(np.arange(len(feat_np))[:200], feat_np[:200, :4],
        #                                 'feature index',
        #                                 'feature value',
        #                                 'features in one clip',
        #                                 outfile, (20, 9),
        #                                 legend=[f'frame {x}' for x in range(len(feat_np))], # ylim=[-0.2, 0.2],
        #                                 mute=True)
        #     # (T, D)
        #     # mean = torch.mean(feat, dim=0, keepdim=True)
        #     for i in range(len(feat)-1):
        #         diff = torch.abs(feat[i]-feat[i+1])
        #         print(f"Frame {i} and {i+1}: feat range: {feat[i].min():.4f}~{feat[i].max():.4f}, avg feat diff: {diff.mean():.4f}, max diff: {diff.max():.4f}")

        # print(t_feat.shape, pose_feat.shape, img_feats.shape, posi.shape)
        if self.add_src_key_mask > 0:
            # mask=True means this key will be ignored
            src_key_mask = img_feats[:, :, -1] < self.add_src_key_mask # for heavy occlusion images, simply do not attend to their features
            # print(src_key_mask[0], img_feats[0, :, -1])
            for i in range(B):
                if torch.sum(src_key_mask[i]) == T:
                    print(f"Warning, all keys are occluded! at example {i}, not using src mask.")
                    src_key_mask[i] = False # setting threshold to 0.35 can cause lots of nan
                    img_feats[i, :, -1] = 0. # set these to zero to indicate it is all occluded
                # print(f"mask sum: {torch.sum(src_key_mask[i])}={T}?")
        else:
            src_key_mask = None

        feats = torch.cat([pose_feat, img_feats, posi, t_feat], -1)
        embed = self.linear_embedding(feats)
        if self.norm == 'batch-emb':
            embed = self.bnorm_emb(embed.permute(0, 2, 1)).permute(0, 2, 1)  # this can also be very large if no bn

        # check the features
        # for i in range(T):
        #     print(
        #         f"Frame {i} ranges: pose {pose_feat[0, i].min():.4f}-{pose_feat[0, i].max():.4f}, img {img_feats[0, i].min():.4f}-{img_feats[0, i].max():.4f}, "
        #         f"positional encoding {posi[0, i].min():.4f}-{posi[0, i].max():.4f}, time {t_feat[0, i].min():.4f}-{t_feat[0, i].max():.4f},"
        #         f"embed {embed[0, i].min():.4f}-{embed[0, i].max()}")

        x = self.encoder(embed, src_key_padding_mask=src_key_mask)
        out = self.decoder(x)
        # print(x.shape, out.shape, t_feat.shape, pose_feat.shape, img_feats.shape, posi.shape)
        return out


class PoseUncertaintyDecoder(nn.Module):
    "a decoder that predicts both the mean and variance of the target"
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        blocks = [
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_dim)
        ]
        self.mean_predictor = nn.Sequential(*blocks)
        blocks = [
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_dim),
            nn.Softplus() # make sure output is always positive
        ]
        self.var_predictor = nn.Sequential(*blocks)

    def forward(self, x):
        """

        Parameters
        ----------
        x : ..., D_embed

        Returns: ..., D_out x 2, first half is mean, the other is variance
        -------

        """
        mu = self.mean_predictor(x)
        var = self.var_predictor(x)
        var = torch.clamp(var, 1e-6, 10)
        return torch.cat([mu, var], -1)



class SMPLCondRotationModelv2(RotationModel):
    "still use MLP, but no activation layer"
    def init_pose_encoder(self, pose_dim, pose_feat_dim):
        ""
        return nn.Sequential(
            nn.Linear(pose_dim, 128),  # pose dimension is 6
            nn.LeakyReLU(),
            nn.Linear(128, pose_feat_dim)
        )


class SMPLRotationWithUncertaintyModel(SMPLCondRotationModelv2):
    def init_decoder(self, embed_dim, out_dim):
        ""
        return PoseUncertaintyDecoder(embed_dim, out_dim)


class SMPLCondRotationModel(RotationModel):
    def init_pose_encoder(self, pose_dim, pose_feat_dim):
        """
        use a transformer to extract SMPL pose information
        Parameters
        ----------
        pose_dim : input dimension
        pose_feat_dim : output pose feature dimension

        Returns
        -------

        """
        linear_embed = nn.Linear(pose_dim, pose_feat_dim) # norm first so no need to append normalization layer
        encoder_layer = nn.TransformerEncoderLayer(pose_feat_dim, nhead=8,
                                                   dim_feedforward=pose_feat_dim,
                                                   batch_first=True,
                                                   activation='gelu')
        pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        return nn.Sequential(linear_embed, pose_encoder)


class SMPLCondCrossAttnQpose(nn.Module):
    """
    given human pose feature + image feature
    do: cross attention with Q=F_pose, K=V=image feature
    then some transformer decoder layer
    and linear output
    """

    def __init__(self, input_dim, embed_dim, d_feedforward=256,
                 pose_dim=6,  # input pose dimension
                 out_dim=6,  # output pose prediction dimension
                 pose_feat_dim=128,
                 num_layers=3, nhead=4,
                 block_dims=[128, 64],
                 norm='none',
                 add_src_key_mask=-1.):
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 128),  # pose dimension is 6
            nn.LeakyReLU(),
            nn.Linear(128, pose_feat_dim)
        ) # input human pose encoder
        self.out_dim = out_dim

        # Batch normalization for features
        img_feat_dim = 768
        self.bnorm_img = nn.BatchNorm1d(img_feat_dim)
        self.bnorm_pose = nn.BatchNorm1d(pose_feat_dim)
        assert norm == 'batch-emb', f'the given normalization {norm} is not supported!'

        # Cross attention layers
        d_model = embed_dim
        self.d_model, self.d_feedforward, self.num_layers = d_model, d_feedforward, num_layers
        self.pose_feat_dim = pose_feat_dim
        cross_attn = nn.TransformerDecoderLayer(d_model=d_model, nhead=8,
                                                dim_feedforward=d_feedforward, batch_first=True,
                                                activation='gelu')
        self.cross_attn_layer = nn.TransformerDecoder(cross_attn, num_layers)

        # project to same feature dimension for cross attention
        self.linear_img = nn.Linear(img_feat_dim, d_model)
        self.linear_pose = nn.Linear(pose_feat_dim, d_model)
        self.bnorm_img_attn = nn.BatchNorm1d(d_model)
        self.bnorm_pose_attn = nn.BatchNorm1d(d_model)

        # Final decoder layer
        self.decoder = nn.Sequential(*[
            nn.Linear(d_model+1, 128), # one additional obj visibility information
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_dim)
        ])

        self.posi_embed = PositionEmbeddingSine_1D(d_model//2, total_feat_dim=d_model) # positional embedding
        self.init_others()

    def init_others(self):
        pass
    def forward(self, pose6d, img_feats, t):
        ""
        B, T = pose6d.shape[:2]
        pose_feat = self.pose_encoder(pose6d)
        # t_feat = self.t_encoder(t)[:, None].repeat(1, T, 1)  # (B,) -> (B, D)
        posi = self.posi_embed(B, T).to(pose6d.device)
        vis = img_feats[:, :, -1:] # object visibility information

        img_feat_norm = self.bnorm_img(img_feats[:, :, :-1].permute(0, 2, 1)).permute(0, 2, 1)
        pose_feat = self.bnorm_pose(pose_feat.permute(0, 2, 1)).permute(0, 2, 1)

        # linear layer so img and pose have same dimension
        img_feat = self.linear_img(img_feat_norm)
        img_feat = self.bnorm_img_attn(img_feat.permute(0, 2, 1)).permute(0, 2, 1) + posi
        pose_feat = self.linear_pose(pose_feat)
        pose_feat = self.bnorm_pose_attn(pose_feat.permute(0, 2, 1)).permute(0, 2, 1) + posi # add positional embedding

        # cross attention layers
        feat = self.cross_attention(img_feat, pose_feat)
        feats = torch.cat([feat, vis], -1)

        out = self.decoder(feats)

        return out

    def cross_attention(self, img_feat, pose_feat):
        feat = self.cross_attn_layer(pose_feat,
                                     img_feat)  # use pose feature to query useful image feature, here Q=pose, K=V=image
        return feat


class SMPLCondCrossAttnQimg(SMPLCondCrossAttnQpose):
    "Q=image feature, K=V=pose feature"
    def cross_attention(self, img_feat, pose_feat):
        """
        exchange the position of img and pose feature
        now Q=image feature, K=V=pose features
        Parameters
        ----------
        img_feat : (B, T, D)
        pose_feat : (B, T, D)

        Returns
        -------

        """
        feat = self.cross_attn_layer(img_feat, pose_feat)  # use pose feature to query useful image feature, here Q=pose, K=V=image
        return feat

class SMPLCondCrossAttnQposeCombine(SMPLCondCrossAttnQpose):
    "feature=Qpose + image feature"
    def init_others(self):
        ""
        # additional encoder layer to aggregate information for: [cross attention, image feature]
        # self attention for image/pose features along, mainly aggregate temporal information
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=8,
                                                   dim_feedforward=self.d_feedforward,
                                                   batch_first=True,
                                                   activation='gelu')  # input (B, L, E)
        self.sa_img = nn.TransformerEncoder(encoder_layer, num_layers=3)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=8,
                                                   dim_feedforward=self.d_feedforward,
                                                   batch_first=True,
                                                   activation='gelu')  # input (B, L, E)
        self.sa_pose = nn.TransformerEncoder(encoder_layer, num_layers=3) # self attention for human pose info

        # change linear model output dim such that we can concatenate the visibility feature into it
        d_model = self.d_model
        self.linear_img = nn.Linear(img_feat_dim, d_model - 1)
        self.linear_pose = nn.Linear(pose_feat_dim, d_model - 1)

        # Final decoder layer
        self.decoder = nn.Sequential(*[
            nn.Linear(d_model + d_model + d_model, 128),  # one additional obj visibility information
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_dim)
        ])

    def forward(self, pose6d, img_feats, t):
        ""
        B, T = pose6d.shape[:2]
        pose_feat = self.pose_encoder(pose6d)
        # t_feat = self.t_encoder(t)[:, None].repeat(1, T, 1)  # (B,) -> (B, D)
        posi = self.posi_embed(B, T).to(pose6d.device)
        vis = img_feats[:, :, -1:] # object visibility information

        img_feat_norm = self.bnorm_img(img_feats[:, :, :-1].permute(0, 2, 1)).permute(0, 2, 1)
        pose_feat = self.bnorm_pose(pose_feat.permute(0, 2, 1)).permute(0, 2, 1)

        # linear layer so img and pose have same dimension
        img_feat = self.linear_img(img_feat_norm)
        img_feat = self.bnorm_img_attn(img_feat.permute(0, 2, 1)).permute(0, 2, 1)
        img_feat = torch.cat([img_feat, vis], -1) + posi # add vis and posi encoding
        pose_feat = self.linear_pose(pose_feat)
        pose_feat = self.bnorm_pose_attn(pose_feat.permute(0, 2, 1)).permute(0, 2, 1)
        pose_feat = torch.cat([pose_feat, vis], -1) + posi

        # self attention for image and pose individually
        pose_feat = self.sa_pose(pose_feat)
        img_feat = self.sa_img(img_feat)

        # cross attention layers, Q=pose feature
        feat = self.cross_attn_layer(pose_feat, img_feat)

        out = self.decoder(torch.cat([img_feat, pose_feat, feat], -1))

        return out

class SMPLCondCrossAttnQimgSelfAttn(SMPLCondCrossAttnQpose):
    "first do self attention on their own features, then do cross attention"
    def init_others(self):
        ""
        # additional encoder layer to aggregate information for: [cross attention, image feature]
        # self attention for image/pose features along, mainly aggregate temporal information
        img_feat_dim = 768
        encoder_layer = nn.TransformerEncoderLayer(img_feat_dim, nhead=8,
                                                   dim_feedforward=self.d_feedforward,
                                                   batch_first=True,
                                                   activation='gelu')  # input (B, L, E)
        self.sa_img = nn.TransformerEncoder(encoder_layer, num_layers=2)
        encoder_layer = nn.TransformerEncoderLayer(self.pose_feat_dim, nhead=8,
                                                   dim_feedforward=self.d_feedforward,
                                                   batch_first=True,
                                                   activation='gelu')  # input (B, L, E)
        self.sa_pose = nn.TransformerEncoder(encoder_layer, num_layers=2) # self attention for human pose info

        # change linear model output dim such that we can concatenate the visibility feature into it
        d_model = self.d_model
        self.linear_img = nn.Linear(img_feat_dim, d_model)
        self.linear_pose = nn.Linear(self.pose_feat_dim, d_model)

        self.posi_embed_img = PositionEmbeddingSine_1D(img_feat_dim // 2, total_feat_dim=img_feat_dim)
        self.posi_embed_pos = PositionEmbeddingSine_1D(self.pose_feat_dim // 2, total_feat_dim=self.pose_feat_dim)

        # Final decoder layer, same as before
        # self.decoder = nn.Sequential(*[
        #     nn.Linear(d_model + 1, 128),  # one additional obj visibility information
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, out_dim)
        # ])
    def forward(self, pose6d, img_feats, t):
        "first self attention then cross attention"
        B, T = pose6d.shape[:2]
        pose_feat = self.pose_encoder(pose6d)
        vis = img_feats[:, :, -1:]  # object visibility information

        posi_img = self.posi_embed_img(B, T).to(pose6d.device)
        img_feat_norm = self.bnorm_img(img_feats[:, :, :-1].permute(0, 2, 1)).permute(0, 2, 1) + posi_img
        posi_pose = self.posi_embed_pos(B, T).to(pose6d.device)
        pose_feat = self.bnorm_pose(pose_feat.permute(0, 2, 1)).permute(0, 2, 1) + posi_pose

        # self attention for image and pose individually
        posi_common = self.posi_embed(B, T).to(pose6d.device)
        pose_feat = self.linear_pose(self.sa_pose(pose_feat))
        pose_feat = self.bnorm_pose_attn(pose_feat.permute(0, 2, 1)).permute(0, 2, 1) + posi_common
        img_feat = self.linear_img(self.sa_img(img_feat_norm))
        img_feat = self.bnorm_img_attn(img_feat.permute(0, 2, 1)).permute(0, 2, 1) + posi_common

        # cross attention, q=image features
        # print(f"Image feature: {torch.max(img_feat[0, 0])}, {torch.min(img_feat[0, 0])}, pose feat: {torch.max(pose_feat[0, 0])},"
        #       f"min={torch.min(pose_feat[0, 0])}")
        feat = self.cross_attn_layer(img_feat, pose_feat)

        out = self.decoder(torch.cat([feat, vis], -1))
        return out

class SMPLCondCrossAttnTwoHead(SMPLCondCrossAttnQpose):
    def init_others(self):
        "additional cross attention + decoder head"
        cross_attn = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8,
                                                dim_feedforward=self.d_feedforward, batch_first=True,
                                                activation='gelu')
        self.cross_attn_layer2 = nn.TransformerDecoder(cross_attn, self.num_layers)
        # additional decoder
        self.decoder2 = nn.Sequential(*[
            nn.Linear(self.d_model + 1, 128),  # one additional obj visibility information
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.out_dim)
        ])

    def forward(self, pose6d, img_feats, t):
        ""
        B, T = pose6d.shape[:2]
        pose_feat = self.pose_encoder(pose6d)
        # t_feat = self.t_encoder(t)[:, None].repeat(1, T, 1)  # (B,) -> (B, D)
        posi = self.posi_embed(B, T).to(pose6d.device)
        vis = img_feats[:, :, -1:]  # object visibility information

        img_feat_norm = self.bnorm_img(img_feats[:, :, :-1].permute(0, 2, 1)).permute(0, 2, 1)
        pose_feat = self.bnorm_pose(pose_feat.permute(0, 2, 1)).permute(0, 2, 1)

        # linear layer so img and pose have same dimension
        img_feat = self.linear_img(img_feat_norm)
        img_feat = self.bnorm_img_attn(img_feat.permute(0, 2, 1)).permute(0, 2, 1) + posi
        pose_feat = self.linear_pose(pose_feat)
        pose_feat = self.bnorm_pose_attn(pose_feat.permute(0, 2, 1)).permute(0, 2, 1) + posi  # add positional embedding

        # cross attention layers
        feat_catt1 = self.cross_attn_layer(pose_feat, img_feat)
        feat_catt2 = self.cross_attn_layer2(img_feat, pose_feat)

        feats1 = torch.cat([feat_catt1, vis], -1)
        feats2 = torch.cat([feat_catt2, vis], -1)

        # two decoder head together to supervise
        out1 = self.decoder(feats1)
        out2 = self.decoder2(feats2)

        out = torch.cat([out1, out2], -1) # (B, D, 12)
        return out
