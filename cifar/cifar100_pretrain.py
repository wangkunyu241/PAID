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
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(384),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize(384),
    transforms.ToTensor(),
])

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def pretrain(description):
    args = load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "cifar100_pretrain":
        logger.info("cifar100 pretrain start")
        model, hra_params = dora_magnitude_angle_hra.configure_model_qkvom_align(base_model, cfg)
        # Prepare dataset
        trainset = torchvision.datasets.CIFAR100(root='/data/bitahub/CIFAR/cifar100', train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR100(root='/data/bitahub/CIFAR/cifar100', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
        
        iter_per_epoch = len(trainloader)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(hra_params, lr=5e-2, momentum=0.9)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
        pretrain_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,8], gamma=0.2)
        
        for epoch in range(1, 10 + 1):
            model.train()
            start = time.time()
            if epoch > 1:
                pretrain_scheduler.step(epoch - 1)
            
            accumulation_steps = 4
            optimizer.zero_grad()

            for batch_index, (images, labels) in enumerate(trainloader):
                labels = labels.cuda()
                images = images.cuda()

                images_chunks = images.chunk(accumulation_steps)
                labels_chunks = labels.chunk(accumulation_steps)

                for img_chunk, lbl_chunk in zip(images_chunks, labels_chunks):
                    outputs = model(img_chunk)
                    loss = loss_function(outputs, lbl_chunk) / accumulation_steps  # 梯度缩放
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * 64 + len(images),
                    total_samples=len(trainloader.dataset)
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
                for (images, labels) in testloader:
                    images = images.cuda()
                    labels = labels.cuda()

                    outputs = model(images)
                    loss = loss_function(outputs, labels)

                    test_loss += loss.item()
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum()

                finish = time.time()
                print('Evaluating Network.....')
                print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                    epoch,
                    test_loss / len(testloader.dataset),
                    correct.float() / len(testloader.dataset),
                    finish - start
                ))

        weights_path = './output/cifar100_pretrain.pth'
        print('saving weights file to {}'.format(weights_path))
        torch.save(model.state_dict(), weights_path)



if __name__ == '__main__':
    pretrain('Cifar100 pretrain.')
