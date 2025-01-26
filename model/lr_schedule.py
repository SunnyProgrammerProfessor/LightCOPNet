# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：lr_schedule.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/3/15 23:43 
"""
import torch.optim as optim


def get_schedule(optimizer, args):
    """Return a learning rate scheduler
        Parameters:
            optimizer          -- the optimizer of the network
            args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                                  opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
        For 'linear', we keep the same learning rate for the first <opt.niter> epochs
        and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        step decay（衰减）会在训练的特定步骤或者周期（Epoch）降低学习率。
        """
    if args.lr_policy == "linear":
        def lambda_rule(epoch):
            lr_rule = 1.0 - epoch / float(args.epochs + 1)
            return lr_rule

        schedule = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == "step":
        step_size = args.epochs // 3
        schedule = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return schedule
