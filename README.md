# PAID: Pairwise Angular-Invariant Decomposition for Continual Test-Time Adaptation
We have released the code for our paper "PAID: Pairwise Angular-Invariant Decomposition for Continual Test-Time Adaptation", which is submitted to NerulPS.

## Prerequisite
[CIFAR10C, CIFAR100C, ImageNetC Datasets Download link] 

https://pan.baidu.com/s/19MJRlaSWDQMYpBF1ppPzhA (pwd: 44t2)

[[CIFAR10-to-CIFAR10C, CIFAR100-to-CIFAR100C, ImageNet-to-ImageNetC Pre-train and Warm-up Checkpoints Download link] 

https://pan.baidu.com/s/1Dz5msymfEZ36_9x807uB6w (pwd: gbbx)

You need to download the dataset and weight archives and extract them into the assets folder under the root directory of PAID. The correct folder structure under assets should be as follows:

```
├── PAID
│   ├── asserts
│   │   ├── CIFAR-10-C
│   │   ├── CIFAR-100-C
│   │   ├── ImageNet-C
│   │   ├── vit_base_patch16_224.augreg_in21k_ft_in1k
│   │   ├── vit_base_patch16_384.augreg_in21k_ft_in1k
│   │   ├── cifar10_warmup.pth
│   │   ├── cifar100_warmup.pth
│   │   ├── imagenet_warmup.pth
...
```

## Classification Experiments
### CIFAR10-to-CIFAR10C-standard task
```bash
# Warmup on CIFAR10
cd PAID/cifar
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
python cifar10_pretrain.py --cfg ./cfgs/cifar10/cifar10_pretrain.yaml

# CTTA on CIFAR10C
cd PAID/cifar
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
python cifar10c_vit_ours.py --cfg cfgs/cifar10/dora_m_a_hra_qkvom_align.yaml --data_dir /data/kunyuwangpKtx/Vida --checkpoint /model/kunyuwangpKtx/Vida/cifar10_warmup.pth
```
### CIFAR100-to-CIFAR100C-standard task
```bash
# Warmup on CIFAR100
cd PAID/cifar
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
python cifar100_pretrain.py --cfg ./cfgs/cifar100/cifar100_pretrain.yaml

# CTTA on CIFAR100C
cd PAID/cifar
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
python cifar100c_vit_ours.py --cfg cfgs/cifar100/dora_m_a_hra_qkvom_align.yaml --data_dir /data/kunyuwangpKtx/Vida  --checkpoint /model/kunyuwangpKtx/Vida/cifar100_warmup.pth
```

### ImageNet-to-ImageNetC-standard task
```bash
# Warmup on ImageNet
cd PAID/imagenet
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
python imagenet_pretrain.py --cfg ./cfgs/vit/imagenet_pretrain.yaml

# CTTA on ImageNetC
cd PAID/imagenet
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
python imagenetc_dora_magnitude_angle_hra.py --cfg ./cfgs/vit/dora_m_a_hra_qkvom_align.yaml --data_dir /data/kunyuwangpKtx/Vida --checkpoint /model/kunyuwangpKtx/Vida/imagenet_warmup.pth
```
