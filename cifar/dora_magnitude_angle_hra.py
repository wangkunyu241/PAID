from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
from time import time
import logging
from torchvision.transforms import InterpolationMode
from typing import List, Tuple
import math
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from robustbench.model_zoo.our_vit import ViT_Ours, ViT_Ours_10
import timm
from tqdm import tqdm
import gc
from torchvision.datasets import VisionDataset, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import numpy as np

logger = logging.getLogger(__name__)

    
class DORA_ALIGN(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, lamda=1.0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "vida requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.lamda = lamda
        self.train_info = None

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs
    
    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        accum_steps = 4
        split_x = torch.chunk(x, accum_steps, dim=0)
        total_output = []

        optimizer.zero_grad()

        for sub_x in split_x:
            sub_x = sub_x.cuda()
            features = self.model.forward_raw_features(sub_x)
            cls_features = features[:, 0]
            batch_std, batch_mean = torch.std_mean(cls_features, dim=0)

            std_loss = torch.norm(batch_std - self.train_info[0].cuda(), p=2)
            mean_loss = torch.norm(batch_mean - self.train_info[1].cuda(), p=2)
            loss = self.lamda * std_loss + mean_loss

            (loss / accum_steps).backward()
            output = self.model.vit.forward_head(features)
            total_output.append(output)

        optimizer.step()
        optimizer.zero_grad()

        return torch.cat(total_output, dim=0)
    
    def obtain_src_stat_cifar10(self, cifar10_root, num_samples=3000):
        net = ViT_Ours_10().eval().cuda()

        class RawCIFAR10Dataset(Dataset):
            def __init__(self, root):
                base = CIFAR10(root=root, train=True, download=False)
                self.data = np.stack([np.array(img) for img, _ in base])  # [N, 32, 32, 3]
                self.labels = [label for _, label in base]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                img = self.data[idx].astype(np.float32) / 255.0  # [H, W, C]
                img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
                return img, self.labels[idx]

        dataset = RawCIFAR10Dataset(cifar10_root)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

        features = []
        num = 0
        with torch.no_grad():
            for images, _ in loader:
                images = F.interpolate(images, size=(384, 384), mode='bilinear', align_corners=False).cuda()
                feat = net.forward_raw_features(images)
                out = net.vit.forward_head(feat)
                ent = -torch.sum(F.softmax(out, dim=1) * F.log_softmax(out, dim=1), dim=1)
                selected = torch.where(ent < math.log(10)/2 - 1)[0]
                features.append(feat[selected][:, 0])
                num += len(selected)
                if num >= num_samples:
                    break

        features = torch.cat(features, dim=0)[:num_samples]
        print(f"[INFO] Source Statistics computed with {features.shape[0]} examples.")
        self.train_info = torch.std_mean(features, dim=0)

        del features
        del net
        gc.collect()
        torch.cuda.empty_cache()
        return self.train_info

    def obtain_src_stat_cifar100(self, cifar100_root, num_samples=3000):
        net = ViT_Ours().eval().cuda()

        class RawCIFAR100Dataset(Dataset):
            def __init__(self, root):
                base = CIFAR100(root=root, train=True, download=False)
                self.data = np.stack([np.array(img) for img, _ in base])  # [N, 32, 32, 3]
                self.labels = [label for _, label in base]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                img = self.data[idx].astype(np.float32) / 255.0  # [H, W, C]
                img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
                return img, self.labels[idx]

        dataset = RawCIFAR100Dataset(cifar100_root)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

        features = []
        num = 0
        with torch.no_grad():
            for images, _ in loader:
                images = F.interpolate(images, size=(384, 384), mode='bilinear', align_corners=False).cuda()
                feat = net.forward_raw_features(images)
                out = net.vit.forward_head(feat)
                ent = -torch.sum(F.softmax(out, dim=1) * F.log_softmax(out, dim=1), dim=1)
                selected = torch.where(ent < math.log(100)/2 - 1)[0]
                features.append(feat[selected][:, 0])
                num += len(selected)
                if num >= num_samples:
                    break

        features = torch.cat(features, dim=0)[:num_samples]
        print(f"[INFO] Source Statistics computed with {features.shape[0]} examples.")
        self.train_info = torch.std_mean(features, dim=0)

        del features
        del net
        gc.collect()
        torch.cuda.empty_cache()
        return self.train_info
    
        
@torch.jit.script
def softmax_entropy_align(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

    
def copy_model_and_optimizer_align(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer_align(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model_qkvom_align(model, cfg):
    for param in model.parameters():
        param.requires_grad = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cpu()
    hra_params, names = inject_hra_qkvom(model.vit, cfg.OPTIM.RANK)
    if cfg.TEST.ckpt != None:
        checkpoint = torch.load(cfg.TEST.ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logger.info('load warmup ckpt successful')
    model.to(device)
    model.train()
    for p in hra_params:
        p.requires_grad = True
    return model, hra_params


class HRAInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, weight=None, bias=None):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features

        m = weight.norm(p=2, dim=1, keepdim=True)
        unit_w = weight / m

        self.m = nn.Parameter(m, requires_grad=True)
        self.unit_w = nn.Parameter(unit_w, requires_grad=False)

        half_u = torch.zeros(in_features, r // 2)
        nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
        self.hra_u = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1), requires_grad=True)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def apply_chain_reflection(self, mat):
        for i in range(self.r):
            u = self.hra_u[:, i].view(-1, 1)
            u = u / u.norm(dim=0, keepdim=True)
            mat = mat - 2 * (mat @ u) @ u.t()
        return mat

    def forward(self, x):
        rotated_unit = self.apply_chain_reflection(self.unit_w)
        final_weight = self.m * rotated_unit
        return F.linear(x, final_weight, self.bias)


def inject_hra_qkvom(model: nn.Module, r: int = 8) -> Tuple[List[nn.Parameter], List[str]]:
    require_grad_params = []
    names = []

    for module_name, module in model.named_modules():
        if module.__class__.__name__ == "Attention" or "Attention" in module_name:
            if hasattr(module, 'qkv') and hasattr(module, 'proj'):
                # 处理 QKV
                qkv = module.qkv
                in_features = qkv.in_features
                out_features = qkv.out_features
                bias = qkv.bias is not None
                qkv_weight = qkv.weight.data.clone()
                qkv_bias = qkv.bias.data.clone() if bias else None

                head_dim = out_features // 3
                q_weight = qkv_weight[:head_dim]
                k_weight = qkv_weight[head_dim:2 * head_dim]
                v_weight = qkv_weight[2 * head_dim:]

                q_bias = qkv_bias[:head_dim] if bias else None
                k_bias = qkv_bias[head_dim:2 * head_dim] if bias else None
                v_bias = qkv_bias[2 * head_dim:] if bias else None

                q_proj = HRAInjectedLinear(in_features, head_dim, r=r, weight=q_weight, bias=q_bias)
                k_proj = HRAInjectedLinear(in_features, head_dim, r=r, weight=k_weight, bias=k_bias)
                v_proj = HRAInjectedLinear(in_features, head_dim, r=r, weight=v_weight, bias=v_bias)

                class HRAQKVWrapper(nn.Module):
                    def __init__(self, q_proj, k_proj, v_proj):
                        super().__init__()
                        self.q = q_proj
                        self.k = k_proj
                        self.v = v_proj

                    def forward(self, x):
                        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)

                module.qkv = HRAQKVWrapper(q_proj, k_proj, v_proj)

                for proj, name_suffix in [(q_proj, "q"), (k_proj, "k"), (v_proj, "v")]:
                    require_grad_params.append(proj.hra_u)
                    names.append(module_name + f".qkv.{name_suffix}.hra_u")

                    require_grad_params.append(proj.m)
                    names.append(module_name + f".qkv.{name_suffix}.m")

                # 处理 attention 的输出投影 proj
                proj = module.proj
                proj_weight = proj.weight.data.clone()
                proj_bias = proj.bias.data.clone() if bias else None

                o_proj = HRAInjectedLinear(proj.in_features, proj.out_features, r=r,
                                           weight=proj_weight, bias=proj_bias)
                module.proj = o_proj

                require_grad_params.append(o_proj.hra_u)
                names.append(module_name + f".proj.hra_u")

                require_grad_params.append(o_proj.m)
                names.append(module_name + f".proj.m")
                
        if module.__class__.__name__ == "Mlp" or "mlp" in module_name:
            for fc_name in ["fc1", "fc2"]:
                if hasattr(module, fc_name):
                    fc = getattr(module, fc_name)
                    if isinstance(fc, nn.Linear):
                        fc_weight = fc.weight.data.clone()
                        fc_bias = fc.bias.data.clone() if fc.bias is not None else None

                        hra_fc = HRAInjectedLinear(fc.in_features, fc.out_features, r=r,
                                                   weight=fc_weight, bias=fc_bias)
                        setattr(module, fc_name, hra_fc)

                        require_grad_params.append(hra_fc.hra_u)
                        names.append(module_name + f".{fc_name}.hra_u")
                        require_grad_params.append(hra_fc.m)
                        names.append(module_name + f".{fc_name}.m")

    return require_grad_params, names

