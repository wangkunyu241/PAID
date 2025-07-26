import torch
import logging
import math

import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import is_model, is_model_in_modules
from timm.models.helpers import load_checkpoint
from timm.models.layers import set_layer_config
from timm.models.hub import load_model_config_from_hf
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import PatchEmbed


logger = logging.getLogger(__name__)


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


import timm
import torch.nn as nn

def modify_head(model):
    model.head = nn.Linear(model.head.in_features, 100)
    return model

class ViT_Ours(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = modify_head(timm.create_model("vit_base_patch16_384", pretrained=False, checkpoint_path="./assets/vit_base_patch16_384.augreg_in21k_ft_in1k/pytorch_model.bin"))
        checkpoint = torch.load("./assets/pretrain_cifar100.t7")
        new_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        self.vit.load_state_dict(new_state_dict, strict=True)
        logger.info('load modify head cifar100 ckpt successful')

    def forward_raw_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward(self, x):
        x = self.forward_raw_features(x)
        x = self.vit.forward_head(x)
        return x
    
    def layers_cls_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    def _collect_layers_features(self, x):
        cls_features = []
        for i in range(len(self.vit.blocks)):
            x = self.vit.blocks[i](x)
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)
        return cls_features

def modify_head_10(model):
    model.head = nn.Linear(model.head.in_features, 10)
    return model

class ViT_Ours_10(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = modify_head_10(timm.create_model("vit_base_patch16_384", pretrained=False, checkpoint_path="./assets/vit_base_patch16_384.augreg_in21k_ft_in1k/pytorch_model.bin"))
        checkpoint = torch.load("./assets/vit_base_384_cifar10.t7")
        new_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        self.vit.load_state_dict(new_state_dict, strict=True)
        logger.info('load modify head cifar10 ckpt successful')

    def forward_raw_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward(self, x):
        x = self.forward_raw_features(x)
        x = self.vit.forward_head(x)
        return x
    
    def layers_cls_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    def _collect_layers_features(self, x):
        cls_features = []
        for i in range(len(self.vit.blocks)):
            x = self.vit.blocks[i](x)
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)
        return cls_features