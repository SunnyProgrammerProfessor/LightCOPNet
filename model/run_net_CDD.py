# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：run_net.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/3/15 22:08 
"""
import os.path
import sys
sys.path.append("/home/data/b532zhaoxiaohui/shuaige/ChangeDetection")

import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from model.lr_schedule import get_schedule
import torch
from model.net_utils import init_net
from model.metric import ConfusionMatrix
from model.logger_record import Logger, Timer
from model import loss
import numpy as np
from model.visualization import make_numpy_grid, vis_norm
import matplotlib.pyplot as plt
from model.MyNet.dual_differ_channelspatial_attention_CDD import MyCDNet
import torch.nn.functional as F
import warnings


def train_model(args, dataloaders, writer):
    warnings.filterwarnings("ignore")
    device = torch.device(args.device)  # device:cuda
    n_class = args.n_class
    batch_size = args.batch_size
    dataloaders = dataloaders  # 数据集
    lr = args.lr  # 学习率
    # args.in_channels = 3
    # args.out_channels = 2
    # args.resnet_stage_num = 5
    # args.if_output_sigmoid = False
    # net = CDBackboneNet(args=args)
    # net = init_net(net=net, gpu_ids=args.gpu_ids, device=device)
    # net = DualEncoderDecoder()
    net = MyCDNet()
    net.to(device)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lr_schedule = get_schedule(optimizer, args)  # 学习率decay策略

    running_metric = ConfusionMatrix(n_class=n_class)  # 计算和存储混淆矩阵，生成模型的各类评价指标(f1,acc,recall,iou etc.)

    # 记录训练的日志
    logger_path = os.path.join(args.checkpoint_root, "log.txt")
    logger = Logger(logger_path)
    logger.write("网络参数: ")
    logger.write_dict_str(args.__dict__)  # 将ArgumentParser中记录的参数打印出来
    logger.write('\n')
    # define time
    timer = Timer()

    # train metrics
    epoch_mf1 = 0
    best_val_mf1 = 0.0
    best_epoch_id = 0
    batch_id = 0  # 当前批次的序号
    epoch_id = 0  # 当前epoch的序号
    epoch_to_start = 0
    # max_epochs = args.epochs
    max_epochs = args.epochs
    global_step = 0  # 当前训练步骤数（在所有训练轮次的总步骤数）
    steps_per_epoch = len(dataloaders["train"])  # 每一个epoch中有多少step， step = (训练数据集大小)/(batch_size)
    total_steps = (max_epochs - epoch_to_start) * steps_per_epoch  # 一共多少steps

    net_pred = None  # 模型的预测输出
    net_loss = None  # 模型的损失
    vis_pred = None  # 预测结果的可视化
    batch = None  # 当前处理的训练批次
    is_training = None  # 当前模型是否为训练模式

    checkpoint_root = args.checkpoint_root  # 日志等文件地址
    vis_root = args.vis_root

    TRAIN_ACC = np.array([], np.float32)  # 记录训练的acc
    if os.path.exists(os.path.join(checkpoint_root, 'train_acc.npy')):
        TRAIN_ACC = np.load(os.path.join(checkpoint_root, 'train_acc.npy'))

    VAL_ACC = np.array([], np.float32)  # 记录验证的acc
    if os.path.exists(os.path.join(checkpoint_root, "val_acc.npy")):
        VAL_ACC = np.load(os.path.join(checkpoint_root, "val_acc.npy"))

    # 定义损失函数
    if args.loss == "ce":
        _pxl_loss = loss.cross_entropy
    elif args.loss == "bce":
        _pxl_loss = loss.binary_ce
    else:
        raise NotImplementedError("Loss function undefined!")

    # ----------------run model--------------------
    # --------------load checkpoints-----------------
    ckpt_name = 'last_ckpt.pt'
    if os.path.exists(os.path.join(args.checkpoint_root, ckpt_name)):
        logger.write("———————loading last checkpoint : ")
        # load the entire checkpoints
        checkpoint = torch.load(os.path.join(args.checkpoint_root, ckpt_name), map_location=device)
        # 加载网络参数
        net.load_state_dict(checkpoint["net_state_dict"])
        # 加载优化器
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # 加载学习率调整方式
        lr_schedule.load_state_dict(checkpoint["lr_schedule_state_dict"])
        net.to(device)  # 主要是预防gpu_ids > 1的情况
        epoch_to_start = checkpoint["epoch_id"] + 1  # 更新开始的epoch
        best_val_mf1 = checkpoint["best_val_mf1"]  # 获取最好的验证集mf1
        best_epoch_id = checkpoint["best_epoch_id"]
        total_steps = (max_epochs - epoch_to_start) * steps_per_epoch
        logger.write(
            f"历史迭代次数:{epoch_to_start - 1}, 历史最优f1 score:{best_val_mf1}(at epoch {best_epoch_id}), 剩余步数:{total_steps}———————\n")
        logger.write("\n")
    # --------------load checkpoints-----------------

    # --------------begin train----------------------
    for epoch_id in range(epoch_to_start, max_epochs):
        # 训练开始之前首先要将混淆矩阵重置
        running_metric.reset()
        # 设置目前状态为训练模式
        is_training = True
        net.train()
        logger.write(f"lr:{optimizer.param_groups[0]['lr']:.7f}\n")
        logger.write('================================Begin Training======================================')
        logger.write("\n")
        for batch_id, batch in enumerate(dataloaders['train']):
            image_A = batch['image_A'].to(device)  # 图片A  8,3,256,256
            image_B = batch['image_B'].to(device)  # 图片B  8,3,256,256
            image_gt = batch['image_Label'].to(device).float()  # 标签  8,1,256,256
            net_pred = net(image_A, image_B)  # 开始训练,forward
            optimizer.zero_grad()  # 清空优化器
            # seg_gt = F.interpolate(batch['image_Label'].float(), scale_factor=1 / 2, mode="bilinear").to(device) #8,1,128,128
            # net_loss = _pxl_loss(net_pred, (image_gt, seg_gt))  # loss
            seg_gt1 = F.interpolate(batch['image_Label'].float(), scale_factor=1 / 2, mode="bilinear").to(
                device)  # 8,1,128,128
            seg_gt2 = F.interpolate(seg_gt1.float(), scale_factor=1 / 2, mode="bilinear").to(device)  # 8,1,64,64
            seg_gt3 = F.interpolate(seg_gt2.float(), scale_factor=1 / 2, mode="bilinear").to(device)  # 8,1,32,32
            net_loss = _pxl_loss(net_pred, (image_gt, seg_gt1, seg_gt2, seg_gt3))
            net_loss.backward()  # backward
            optimizer.step()
            # collect running batch states 统计每个batch的训练情况
            gt_detach = batch['image_Label'].to(device).detach()  # dtype:uint8 only image_Label
            # pred_detach = net_pred.detach()
            # pred_detach = torch.argmax(pred_detach, dim=1)
            pred_detach = torch.sigmoid(net_pred.squeeze(1).detach())
            pred_detach = (pred_detach >= 0.5).int()
            running_mf1 = running_metric.update_cm(pr=pred_detach.cpu().numpy(), gt=gt_detach.squeeze().cpu().numpy(),
                                                   weight=1)  # train:mean_f1
            m = len(dataloaders['train'])  # train steps
            # 更新训练时间
            global_step = (epoch_id - epoch_to_start) * steps_per_epoch + batch_id
            timer.update_progress((global_step + 1) / total_steps)  # 更新训练的进度
            remain_time = timer.estimated_remaining()  # 剩余训练时间
            per_step_time = (global_step + 1) * batch_size / timer.get_stage_elapsed()  # 每一个dual-image训练的时间
            if np.mod(batch_id, 10) == 1:  # 每隔10steps输出一次
                message = f"Is_Training :{is_training}, 当前训练epoch:{epoch_id + 1}/{max_epochs}" \
                          f", 当前训练batch:{batch_id}/{m}, 当前平均f1 score:{running_mf1}, 总计batch训练时间:{per_step_time * batch_size}" \
                          f", 预计剩余训练时间:{remain_time}h, loss:{net_loss.item()} "
                logger.write(message=message)
                logger.write("\n")
            if np.mod(batch_id, 1000) == 1:  # LEVIR-CD数据集batch_size=16时,batch_id<500,因此每一个epoch可视化一次结果
                vis_imageA = make_numpy_grid(vis_norm(batch['image_A']))  # imageA的原始图像
                vis_imageB = make_numpy_grid(vis_norm(batch['image_B']))  # imageB的原始图像
                # vis_pred = make_numpy_grid(
                #     torch.argmax(net_pred, dim=1, keepdim=True) * 255)  # 预测结果（*255有点问题,需要softmax类似的操作）
                vis_pred = make_numpy_grid(pred_detach.unsqueeze(1))
                vis_gt = make_numpy_grid(batch['image_Label'])
                FN_mask = (vis_gt[:, :, 0] == 1) & (vis_pred[:, :, 0] == 0)  # FN
                FP_mask = (vis_gt[:, :, 0] == 0) & (vis_pred[:, :, 0] == 1)  # FP
                TN_mask = (vis_gt[:, :, 0] == 0) & (vis_pred[:, :, 0] == 0)  # TN
                TP_mask = (vis_gt[:, :, 0] == 1) & (vis_pred[:, :, 0] == 1)  # TP
                vis_change = np.zeros_like(vis_pred)
                # FN用蓝色表示[0,0,1] ,FP用红色表示[1,0,0] ,TN用黑色表示[0,0,0],TP用白色表示[1,1,1]
                vis_change[:, :, 0][FN_mask] = 0
                vis_change[:, :, 1][FN_mask] = 0
                vis_change[:, :, 2][FN_mask] = 1

                vis_change[:, :, 0][FP_mask] = 1
                vis_change[:, :, 1][FP_mask] = 0
                vis_change[:, :, 2][FP_mask] = 0

                vis_change[:, :, 0][TP_mask] = 1
                vis_change[:, :, 1][TP_mask] = 1
                vis_change[:, :, 2][TP_mask] = 1

                vis_change[:, :, 0][TN_mask] = 0
                vis_change[:, :, 1][TN_mask] = 0
                vis_change[:, :, 2][TN_mask] = 0
                vis = np.concatenate([vis_imageA, vis_imageB, vis_pred, vis_gt, vis_change], axis=0)
                vis = np.clip(vis, a_min=0.0, a_max=1.0)
                vis_filename = os.path.join(vis_root, "isTrain_" + str(is_training) + "_" + "epoch_" + str(
                    epoch_id + 1) + "_" + "batch_" + str(batch_id) + "_" + ".jpg")
                plt.imsave(vis_filename, vis)

        # collect epoch states  统计每个epoch的训练情况
        scores = running_metric.get_scores()
        epoch_mf1 = scores["mf1"]
        logger.write(f"Epoch:{epoch_id + 1}/{max_epochs}, epoch_meanf1:{epoch_mf1}, ")
        message_epoch = ""
        for k, v in scores.items():
            message_epoch += f"{k}: {v:.5f} "
        logger.write(message_epoch + '\n')
        logger.write("\n")
        logger.write("\n")
        writer.add_scalar("train/acc", scores["acc"], epoch_id + 1)
        writer.add_scalar("train/mf1", epoch_mf1, epoch_id + 1)
        writer.add_scalar("train/iou", scores["iou_0"], epoch_id + 1)
        writer.add_scalar("train/recall", scores["recall_0"], epoch_id + 1)
        writer.add_scalar("train/precision", scores["precision_0"], epoch_id + 1)

        # 更新TRAIN_ACC
        TRAIN_ACC = np.append(TRAIN_ACC, [epoch_mf1])
        np.save(os.path.join(checkpoint_root, 'train_acc.npy'), TRAIN_ACC)

        #  更新lr策略
        lr_schedule.step()
        # ----------------end train----------------------

        # ----------------begin val----------------------
        running_metric.reset()
        is_training = False
        net.eval()
        logger.write("================================Begin verification======================================")
        logger.write("\n")
        for batch_id, batch in enumerate(dataloaders['val']):
            with torch.no_grad():
                image_A = batch['image_A'].to(device)
                image_B = batch['image_B'].to(device)
                net_pred = net(image_A, image_B)
            # collect running batch states 统计每个batch的训练情况
            gt_detach = batch['image_Label'].to(device).detach()
            # pred_detach = net_pred.detach()
            # pred_detach = torch.argmax(pred_detach, dim=1)
            pred_detach = torch.sigmoid(net_pred.squeeze(1).detach())
            pred_detach = (pred_detach >= 0.5).int()
            running_mf1 = running_metric.update_cm(pr=pred_detach.cpu().numpy(),
                                                   gt=gt_detach.squeeze().cpu().numpy())  # val:mean_f1
            m = len(dataloaders['val'])
            global_step = (epoch_id - epoch_to_start) * m + batch_id
            timer.update_progress((global_step + 1) / ((max_epochs - 0) / m))
            remain_time = timer.estimated_remaining()
            per_step_time = (global_step + 1) * batch_size / timer.get_stage_elapsed()  # 每一个dual-image验证的时间
            if np.mod(batch_id, 10) == 1:
                message = f"Is_Training:{is_training}, 当前训练Epoch:{epoch_id + 1}/{max_epochs}, " \
                          f"当前训练batch:{batch_id}/{m}, 当前平均f1 score:{running_mf1}, 每一个batch验证的时间:{per_step_time * batch_size}, " \
                          f"预计剩余验证时间:{remain_time} \n "
                logger.write(message=message)
            if np.mod(batch_id, 500) == 1:
                vis_imageA = make_numpy_grid(vis_norm(batch['image_A']))
                vis_imageB = make_numpy_grid(vis_norm(batch['image_B']))
                # vis_pred = make_numpy_grid(torch.argmax(net_pred, dim=1, keepdim=True) * 255)
                vis_pred = make_numpy_grid(pred_detach.unsqueeze(1))
                vis_gt = make_numpy_grid(batch['image_Label'])
                FN_mask = (vis_gt[:, :, 0] == 1) & (vis_pred[:, :, 0] == 0)  # FN
                FP_mask = (vis_gt[:, :, 0] == 0) & (vis_pred[:, :, 0] == 1)  # FP
                TN_mask = (vis_gt[:, :, 0] == 0) & (vis_pred[:, :, 0] == 0)  # TN
                TP_mask = (vis_gt[:, :, 0] == 1) & (vis_pred[:, :, 0] == 1)  # TP
                vis_change = np.zeros_like(vis_pred)
                # FN用蓝色表示[0,0,1] ,FP用红色表示[1,0,0] ,TN用黑色表示[0,0,0],TP用白色表示[1,1,1]
                vis_change[:, :, 0][FN_mask] = 0
                vis_change[:, :, 1][FN_mask] = 0
                vis_change[:, :, 2][FN_mask] = 1

                vis_change[:, :, 0][FP_mask] = 1
                vis_change[:, :, 1][FP_mask] = 0
                vis_change[:, :, 2][FP_mask] = 0

                vis_change[:, :, 0][TP_mask] = 1
                vis_change[:, :, 1][TP_mask] = 1
                vis_change[:, :, 2][TP_mask] = 1

                vis_change[:, :, 0][TN_mask] = 0
                vis_change[:, :, 1][TN_mask] = 0
                vis_change[:, :, 2][TN_mask] = 0
                vis = np.concatenate([vis_imageA, vis_imageB, vis_pred, vis_gt, vis_change], axis=0)
                vis = np.clip(vis, a_min=0.0, a_max=1.0)
                vis_filename = os.path.join(vis_root, "isTrain_" + str(is_training) + "-" + "epoch_" + str(
                    epoch_id + 1) + "_" + "batch_" + str(batch_id) + "_" + ".jpg")
                plt.imsave(vis_filename, vis)
        scores = running_metric.get_scores()
        epoch_mf1 = scores["mf1"]
        logger.write(f"is_Training:{is_training}, Epoch:{epoch_id + 1}/{max_epochs}, epoch_meanf1:{epoch_mf1}, ")
        message_epoch = ''
        for k, v in scores.items():
            message_epoch += f"{k}: {v:.5f} "
        logger.write(message_epoch + "\n")
        logger.write("\n")
        logger.write("\n")
        writer.add_scalar("val/acc", scores["acc"], epoch_id + 1)
        writer.add_scalar("val/mf1", epoch_mf1, epoch_id + 1)
        writer.add_scalar("val/iou", scores["iou_0"], epoch_id + 1)
        writer.add_scalar("val/recall", scores["recall_0"], epoch_id + 1)
        writer.add_scalar("val/precision", scores["precision_0"], epoch_id + 1)

        # 更新VAL_ACC
        VAL_ACC = np.append(VAL_ACC, [epoch_mf1])
        np.save(os.path.join(checkpoint_root, "val_acc.npy"), VAL_ACC)
        # -----------------end val-----------------------

        # --------------save checkpoints-----------------
        if os.path.exists(os.path.join(checkpoint_root, "best_ckpt.pt")):
            checkpoint = torch.load(os.path.join(checkpoint_root, "best_ckpt.pt"), map_location=device)
            if epoch_mf1 >= checkpoint["best_val_mf1"]:
                best_val_mf1 = epoch_mf1
                best_epoch_id = epoch_id + 1
            else:
                best_val_mf1 = checkpoint["best_val_mf1"]
                best_epoch_id = checkpoint["best_epoch_id"]
        else:
            best_val_mf1 = epoch_mf1
            best_epoch_id = epoch_id + 1
        torch.save({
            "epoch_id": epoch_id + 1,
            "best_val_mf1": best_val_mf1,
            "best_epoch_id": best_epoch_id,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_schedule_state_dict": lr_schedule.state_dict()
        }, os.path.join(args.checkpoint_root, ckpt_name))
        logger.write(
            f"Latest model updated ! Epoch_f1(val):{epoch_mf1}, Historical_Epoch_id:{best_epoch_id}, Best_Val_f1:{best_val_mf1}")
        logger.write('\n')
        # --------------save checkpoints-----------------

        # --------------update checkpoints-----------------
        if epoch_mf1 >= best_val_mf1:
            best_val_mf1 = epoch_mf1
            best_epoch_id = epoch_id + 1
            torch.save({
                "epoch_id": epoch_id + 1,
                "best_val_mf1": best_val_mf1,
                "best_epoch_id": best_epoch_id,
                "net_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_schedule_state_dict": lr_schedule.state_dict()
            }, os.path.join(checkpoint_root, "best_ckpt.pt"))
            logger.write("***************************Best model updated ! ***************************\n")
            logger.write(f"Best f1 score at present:{best_val_mf1}(at epoch{best_epoch_id})")
            logger.write("\n")
        # else:
        #     checkpoint = torch.load(os.path.join(checkpoint_root, "best_ckpt.pt"), map_location=device)
        #     best_val_acc = checkpoint["best_val_acc"]
        #     best_epoch_id = checkpoint["best_epoch_id"]
        #     logger.write(
        #         f"Latest model updated ! Epoch_acc(val):{epoch_acc},Historical_Epoch_id:{best_epoch_id},Best_Val_acc:{best_val_acc}")
        #     logger.write('\n')
        # --------------update checkpoints-----------------
    writer.close()
