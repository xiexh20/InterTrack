import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from timm.models.vision_transformer import VisionTransformer, resize_pos_embed
from torch import Tensor
from torchvision.transforms import functional as TVF


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

MODEL_URLS = {
    'vit_base_patch16_224_mae': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    'vit_small_patch16_224_msn': 'https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar',
    'vit_large_patch7_224_msn': 'https://dl.fbaipublicfiles.com/msn/vitl7_200ep.pth.tar',
}

NORMALIZATION = {
    'vit_base_patch16_224_mae': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'vit_small_patch16_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'vit_large_patch7_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'dinov2_vitb14': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'dinov2_vitb14_tune': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
}

MODEL_KWARGS = {
    'vit_base_patch16_224_mae': dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
    ), 
    'vit_small_patch16_224_msn': dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
    ),
    'vit_large_patch7_224_msn': dict(
        patch_size=7, embed_dim=1024, depth=24, num_heads=16,
    )
}


class Encoder16x16(nn.Module):
    "takes dino Dx16x16 feature as input, and output D_out feature vector"
    def __init__(self, cin, cout, nf=256, activation=None):
        super().__init__()
        network = [
            # nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            # nn.GroupNorm(nf//4, nf),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
        ]
        # if activation is not None:
        #     network += [get_activation(activation)]
        assert activation is None
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)

class FeatureModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self, 
        image_size: int = 224,
        model_name: str = 'vit_small_patch16_224_mae',
        global_pool: str = '',  # '' or 'token'
    ) -> None:
        super().__init__()
        self.model_name = model_name

        # Identity
        if self.model_name == 'identity':
            return
        self.mean, self.std = NORMALIZATION[model_name]


        # # Modify MSN model with output head from training
        # if model_name.endswith('msn'):
        #     use_bn = True
        #     emb_dim = (192 if 'tiny' in model_name else 384 if 'small' in model_name else 
        #         768 if 'base' in model_name else 1024 if 'large' in model_name else 1280)
        #     hidden_dim = 2048
        #     output_dim = 256
        #     self.model.fc = None
        #     fc = OrderedDict([])
        #     fc['fc1'] = torch.nn.Linear(emb_dim, hidden_dim)
        #     if use_bn:
        #         fc['bn1'] = torch.nn.BatchNorm1d(hidden_dim)
        #     fc['gelu1'] = torch.nn.GELU()
        #     fc['fc2'] = torch.nn.Linear(hidden_dim, hidden_dim)
        #     if use_bn:
        #         fc['bn2'] = torch.nn.BatchNorm1d(hidden_dim)
        #     fc['gelu2'] = torch.nn.GELU()
        #     fc['fc3'] = torch.nn.Linear(hidden_dim, output_dim)
        #     self.model.fc = torch.nn.Sequential(fc)
        if model_name == 'dinov2_vitb14':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to('cuda')
        elif model_name == 'dinov2_vitb14_tune':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to('cuda')
            self.feat_fuser = Encoder16x16(self.model.embed_dim, self.model.embed_dim, self.model.embed_dim) # fuse 16x16 to a vector
        else:
            # Old models used by pc2
            # Create model
            self.model = VisionTransformer(
                img_size=image_size, num_classes=0, global_pool=global_pool,
                **MODEL_KWARGS[model_name])
            # Load pretrained checkpoint
            checkpoint = torch.hub.load_state_dict_from_url(MODEL_URLS[model_name])
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'target_encoder' in checkpoint:
                state_dict = checkpoint['target_encoder']
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                # NOTE: Comment the line below if using the projection head, uncomment if not using it
                # See https://github.com/facebookresearch/msn/blob/81cb855006f41cd993fbaad4b6a6efbb486488e6/src/msn_train.py#L490-L502
                # for more info about the projection head
                state_dict = {k: v for k, v in state_dict.items() if not k         .startswith('fc.')}
            else:
                raise NotImplementedError()
            state_dict['pos_embed'] = resize_pos_embed(state_dict['pos_embed'], self.model.pos_embed)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        # Model properties
        self.feature_dim = self.model.embed_dim

        # # Modify MSN model with output head from training
        # if model_name.endswith('msn'):
        #     self.fc = self.model.fc
        #     del self.model.fc
        # else:
        #     self.fc = nn.Identity()
        
        # NOTE: I've disabled the whole projection head stuff for simplicity for now
        self.fc = nn.Identity()

    def denormalize(self, img: Tensor):
        "(x-(-m/s))/(1/s)=sx + m: 0-1-> "
        img = TVF.normalize(img, mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std])
        return torch.clip(img, 0, 1) # does img need to be between 0-1?

    def normalize(self, img: Tensor):
        return TVF.normalize(img, mean=self.mean, std=self.std)

    def forward(
        self, 
        x: Tensor, 
        return_type: str = 'features',
        return_upscaled_features: bool = True,
        return_projection_head_output: bool = False,
    ):
        """Normalizes the input `x` and runs it through `model` to obtain features"""
        assert return_type in {'cls_token', 'features', 'all', 'feat_avg'}

        # Identity
        if self.model_name == 'identity':
            return x
        # from implicitron.dataset.utils.py L231, the raw images are divided by 255
        # Co3dv2 dataset: Input image range: tensor(0., device='cuda:0') tensor(0.9961, device='cuda:0')
        # print("Input image range:", torch.min(x), torch.max(x))
        # Normalize and forward
        B, C, H, W = x.shape
        x = self.normalize(x)
        with torch.no_grad():
            if self.model_name in ['dinov2_vitb14', 'dinov2_vitb14_tune']:
                out = self.model.forward_features(x)
                feats = torch.cat([out['x_norm_clstoken'][:, None], out['x_norm_patchtokens']], 1)
            else:
                feats = self.model(x) # B, T, D

        # Reshape to image-like size
        if return_type in {'features', 'all', 'feat_avg'}:
            B, T, D = feats.shape
            assert math.sqrt(T - 1).is_integer()
            HW_down = int(math.sqrt(T - 1))  # subtract one for CLS token
            output_feats: Tensor = feats[:, 1:, :].reshape(B, HW_down, HW_down, D).permute(0, 3, 1, 2)  # (B, D, H_down, W_down)
            # feature shape: torch.Size([16, 384, 14, 14]) input shape: torch.Size([16, 3, 224, 224])
            # feature shape: torch.Size([16, 384, 32, 32]) input shape: torch.Size([16, 3, 512, 512]
            # print("Image feature shape:", output_feats.shape, "input shape:", x.shape)
            if return_upscaled_features:
                output_feats = F.interpolate(output_feats, size=(H, W), mode='bilinear',
                    align_corners=False)  # (B, D, H_orig, W_orig) XH: why do this??? it is memory intensive!

        # Head for MSN
        output_cls = feats[:, 0] # what is cls token?
        if return_projection_head_output and return_type in {'cls_token', 'all'}:
            output_cls = self.fc(output_cls)

        # MAE model: cls_token shape: torch.Size([16, 768]), output_feats shape: torch.Size([16, 768, 224, 224])
        # print(f"cls_token shape: {output_cls.shape}, output_feats shape: {output_feats.shape}")
        # Return
        if return_type == 'cls_token':
            if self.model_name == 'dinov2_vitb14_tune':
                B, T, D = feats.shape
                HW_down = int(math.sqrt(T - 1))
                feat_out = feats[:, 1:, :].reshape(B, HW_down, HW_down, D).permute(0, 3, 1, 2) # (B, D, H, W)
                cls_fuse = self.feat_fuser(feat_out) # (B, D)
                return cls_fuse
            return output_cls
        elif return_type == 'features':
            return output_feats
        elif return_type == 'feat_avg':
            return output_feats.mean(dim=(-2, -1))
        else:
            return output_cls, output_feats
