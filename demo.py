import argparse
import os
import cv2
import torch
import numpy as np
from core.utils import load_cfg, load_weights, tensor_to_img
from core.distiller import Distiller
from core.model_zoo import model_zoo
import pdb

def main(args):
    cfg = load_cfg(args.cfg)
    # (Pdb) pp cfg
    # {'distillation_loss': {'generator_weights': [1.0, 1.0, 1.0, 0.5, 0.1],
    #                        'perceptual_size': 256},
    #  'logger': {'params': {'experiment_name': 'baseline',
    #                        'offline_mode': True,
    #                        'project_name': 'bes-dev/stylegan2_compression'},
    #             'type': 'NeptuneLogger'},
    #  'teacher': {'mapping_network': {'name': 'stylegan2_ffhq_config_f_mapping_network.ckpt'},
    #              'synthesis_network': {'name': 'stylegan2_ffhq_config_f_synthesis_network.ckpt'}},
    #  'trainer': {'batch_size': 2,
    #              'lr_gan': 0.0005,
    #              'lr_student': 0.0005,
    #              'max_epochs': 100,
    #              'mode': 'g,d',
    #              'monitor': 'kid_val',
    #              'monitor_mode': 'min',
    #              'num_workers': 0,
    #              'style_mean': 4096,
    #              'style_mean_weight': 0.5},
    #  'trainset': {'emb_size': 512, 'n_batches': 10000},
    #  'valset': {'emb_size': 512, 'n_batches': 200}}
    distiller = Distiller(cfg)
    if args.ckpt is not None:
        ckpt = model_zoo(args.ckpt)
        load_weights(distiller, ckpt["state_dict"])
        print("model_load(distiller, {})".format(args.ckpt))

    # torch.save(distiller.mapping_net.state_dict(), "models/FaceMap.pth")
    # torch.save(distiller.student.state_dict(), "models/FaceGen.pth")

    while True:
        var = torch.randn(1, distiller.mapping_net.style_dim)
        img_s = distiller(var, truncated=args.truncated)
        cv2.imshow("demo", tensor_to_img(img_s[0].cpu()))

        key = chr(cv2.waitKey(30) & 255)

        if key == 'q':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # pipeline configure
    parser.add_argument("--cfg", type=str, default="configs/mobile_stylegan_ffhq.json", help="path to config file")
    parser.add_argument("--ckpt", type=str, default="models/mobilestylegan_ffhq.ckpt", help="path to checkpoint")
    parser.add_argument("--truncated", action='store_true', help="use truncation mode")
    args = parser.parse_args()
    main(args)
