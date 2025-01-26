# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/4/26 14:46 
"""
import torch
import os
import sys
sys.path.append("/home/data/b532zhaoxiaohui/shuaige/ChangeDetection")
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter

from datasets.datasets_config import CDDataset
from torch.utils.data import DataLoader
# from model.run_net_1 import train_model
from model.run_net_ablation_SkipFusionEnhance import train_model
import warnings
"""
    消融实验: Skip Fusion Enhancemnet
"""


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0", help="e.g 0  0,1,2  or use -1 for cpu")
    parser.add_argument("--device", type=str, default="cuda", help="choose your cpu or gpu")
    parser.add_argument("--form", type=str, default="train", help="train or test")
    parser.add_argument("--image_path", type=str, default="/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/LEVIR/list",
                        help="directory for storing datasets")
    parser.add_argument("--train_val", type=str, default="val", help="Verify during training!")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="CDDataset", help="choose your dataset and transform")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--is_transform", default=True, type=bool)
    parser.add_argument("--checkpoint_root", default="/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/checkpoints/MyNet/third", type=str)
    parser.add_argument("--vis_root", default="/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/third", type=str)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_policy", type=str, default="linear")
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--loss", default='bce', type=str)
   

    args = parser.parse_args()
    assert (torch.cuda.is_available())
    # checkpoints root: save model
    if not os.path.exists(args.checkpoint_root):
        os.mkdir(args.checkpoint_root)
    # vis_dir root : Visualization results
    if not os.path.exists(args.vis_root):
        os.mkdir(args.vis_root)

    train_dataset = CDDataset(path=args.image_path, form=args.form, img_size=args.image_size,
                              transform=args.is_transform,
                              is_train=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=True)
    val_dataset = CDDataset(path=args.image_path, form=args.train_val, img_size=args.image_size,
                            transform=args.is_transform,
                            is_train=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=0,
                                shuffle=False)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    writer = SummaryWriter("/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/logs/MyNet/third")
    train_model(args, dataloaders, writer)
    writer.close()