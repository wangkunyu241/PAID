import logging

import torch
import torch.optim as optim
import math

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy


from conf import cfg, load_cfg_fom_args
import dora_magnitude_angle_hra
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from PIL import Image


logger = logging.getLogger(__name__)

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class SafeImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, IOError, Image.DecompressionBombError) as e:
            print(f"[WARN] Skipped corrupted image at index {index}: {e}")
            return self.__getitem__((index + 1) % len(self))

def get_dataloader_imagenet(datapath, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.4),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0))], p=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
         ])
    # 加载训练集
    train_dataset = SafeImageFolder(root=datapath+"/train",transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 加载验证集
    val_dataset = SafeImageFolder(root=datapath+"/val",transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return train_loader, val_loader

def pretrain(description):
    args = load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "imagenet_pretrain":
        logger.info("imagenet pretrain start")
        model, hra_params = dora_magnitude_angle_hra.configure_model_qkvom(base_model, cfg)
        training_loader, test_loader = get_dataloader_imagenet(
            '/data/linshiqi047/imagenet',
            num_workers=4,
            batch_size=cfg.BATCH_SIZE,
        )
        iter_per_epoch = len(training_loader)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(hra_params, lr=cfg.PRETRAIN_LR, momentum=0.9)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
        pretrain_scheduler = CosineAnnealingLR(optimizer, T_max=5)
        for epoch in range(1, 3 + 1):
            model.train()
            start = time.time()
            if epoch > 1:
                pretrain_scheduler.step(epoch-1)
            for batch_index, (images, labels) in enumerate(training_loader):
                labels = labels.cuda()
                images = images.cuda()

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * cfg.BATCH_SIZE + len(images),
                    total_samples=len(training_loader.dataset)
                ))

                if epoch <= 1:
                   warmup_scheduler.step()

            finish = time.time()
            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

            with torch.no_grad():
                model.eval()
                test_loss = 0.0
                correct = 0.0
                start = time.time()
                for (images, labels) in test_loader:
                    images = images.cuda()
                    labels = labels.cuda()

                    outputs = model(images)
                    loss = loss_function(outputs, labels)

                    test_loss += loss.item()
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum()

                finish = time.time()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')
                print('Evaluating Network.....')
                print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                    epoch,
                    test_loss / len(test_loader.dataset),
                    correct.float() / len(test_loader.dataset),
                    finish - start
                ))

        weights_path = '/output/imagenet_pretrain.pth'
        print('saving weights file to {}'.format(weights_path))
        torch.save(model.state_dict(), weights_path)
    


if __name__ == '__main__':
    pretrain('"Imagenet pretrain.')
