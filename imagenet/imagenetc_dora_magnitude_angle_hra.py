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
import numpy as np


logger = logging.getLogger(__name__)
args = load_cfg_fom_args("auto-tune")
train_info = None


def evaluate():
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

    if cfg.MODEL.ADAPTATION == "dora_magnitude_angle_hra_qkvom_align":
        logger.info("test-time adaptation: dora_magnitude_angle_hra_align")
        model = setup_dora_qkvom_align(args, base_model)

    prev_ct = "x0"
    All_error = []
    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            try:
                if i_x == 0:
                    model.reset()
                    logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            err = 1. - acc
            All_error.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
    return All_error


def setup_optimizer(params):
    
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    elif cfg.OPTIM.METHOD == 'AdamW':
        return optim.AdamW(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError


def setup_dora_qkvom_align(args, model):
    global train_info
    model, dora_param = dora_magnitude_angle_hra.configure_model_qkvom_align(model, cfg)
    optimizer = setup_optimizer(dora_param)
    dora_model = dora_magnitude_angle_hra.DORA_ALIGN(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           lamda=cfg.OPTIM.LAMDA
                           )

    if train_info is not None:
        dora_model.train_info = train_info
        logger.info("Loaded existing global train_info.")
    else:
        train_info_capture = dora_model.obtain_src_stat(
            data_path='/data/linshiqi047/imagenet',
            num_samples=3000
        )
        train_info = train_info_capture
        logger.info("Obtained new train_info and saved globally.")

    return dora_model


if __name__ == "__main__":
    evaluate()
