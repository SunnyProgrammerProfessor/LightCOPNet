# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：metric.py   模型评价指标
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/3/20 22:42 
"""
import numpy as np


class Average(object):
    # 计算和存储平均值和当前值
    def __init__(self):
        self.initialized = False  # 判断指标是否已经初始化
        self.current_value = None  # 当前值
        self.average_value = None  # 平均值
        self.sum = None  # 计算和
        self.count = None  # 计算数量

    def initialize(self, current_value, weight):
        self.current_value = current_value
        self.average_value = current_value
        self.sum = current_value * weight
        self.count = weight
        self.initialized = True

    def add(self, current_value, weight):
        self.current_value = current_value
        self.sum += current_value * weight
        self.count += weight
        self.average_value = self.sum / self.count

    def update(self, current_value, weight=1):
        if not self.initialized:
            self.initialize(current_value, weight)
        else:
            self.add(current_value, weight)

    def get_value(self):
        return self.current_value

    def get_average(self):
        return self.average_value

    def get_sum(self):
        return self.sum

    def get_count(self):
        return self.count

    def get_scores(self):
        scores = metric_scores(self.sum)  # 获取f1，recall，acc，IoU等各类指标的得分
        return scores

    def reset(self):  # 重置
        self.initialized = False


class ConfusionMatrix(Average):
    def __init__(self, n_class):
        super(ConfusionMatrix, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):  # pr:预测的标签  gt:真实的标签
        # 获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵
        current_value = get_confusion_matrix(n_class=self.n_class, pr=pr, gt=gt)  # 获得当前的混淆矩阵:ndarray(2,2)
        self.update(current_value, weight)
        current_score = get_current_f1(current_value)  # 计算当前f1得分
        return current_score

    def get_scores(self):
        scores = metric_scores(self.sum)
        return scores


def get_confusion_matrix(n_class, pr, gt):
    confusion_matrix = np.zeros((n_class, n_class))

    def compare(pr, gt):
        mask = (gt >= 0) & (gt < n_class)  # 过滤掉无效的标签值
        # 这里比较精妙的，原理是n_class*gt+pr的结果有四种情况,TN=0,FP=1,FN=2,TP=3
        # 理论上hist的排序是TN,FP,FN,TP与混淆矩阵是反的,但是,metric_scores函数返回的2分类指标结果的顺序是[0,1],因此hist不需要反转
        hist = np.bincount(n_class * gt[mask].astype(int) + pr[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    for lt, lp in zip(gt, pr):
        confusion_matrix += compare(lt.flatten(), lp.flatten())  # lt.flatten() : ndarray(65536,)
    return confusion_matrix


def get_current_f1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)  # 真阳性和真阴性:ndarray(2,)
    sum_h = hist.sum(axis=1)  # 每行的总和，某一类的实际数量之和:ndarray(2,)
    sum_v = hist.sum(axis=0)  # 每列的综合，某一类的预测数量之和:ndarray(2,)
    bias = np.finfo(np.float32).eps
    # 1.计算acc
    acc = tp.sum() / (hist.sum() + bias)
    # 2.计算recall
    recall = tp / (sum_v + bias)  # recall:ndarray(2,)
    # 3.计算precision
    precision = tp / (sum_h + bias)  # precision:ndarray(2,)
    # 4.F1 score
    f1 = 2 * recall * precision / (recall + precision + bias)  # f1:ndarray(2,)
    mean_f1 = np.nanmean(f1)  # 计算平均的F1  score
    return mean_f1


def metric_scores(confusion_matrix):  # 模型的所有指标
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)  # 真阳性和真阴性
    sum_h = hist.sum(axis=1)  # 每行的总和，某一类的实际数量之和
    sum_v = hist.sum(axis=0)  # 每列的综合，某一类的预测数量之和
    bias = np.finfo(np.float32).eps
    # 1. 计算acc
    acc = tp.sum() / (hist.sum() + bias)
    # 2.计算recall
    recall = tp / (sum_v + bias)
    cls_recall = dict(zip(["recall_" + str(i) for i in range(n_class)], recall))
    # 3.计算precision
    precision = tp / (sum_h + bias)
    cls_precision = dict(zip(["precision_" + str(i) for i in range(n_class)], precision))
    # 4.F1 score
    f1 = 2 * recall * precision / (recall + precision + bias)
    cls_f1 = dict(zip(["f1_" + str(i) for i in range(n_class)], f1))
    mean_f1 = np.nanmean(f1)  # 计算平均的F1  score
    # 5.计算IoU
    IoU = tp / (sum_h + sum_v - tp + bias)
    cls_iou = dict(zip(["iou_" + str(i) for i in range(n_class)], IoU))
    mean_IoU = np.nanmean(IoU)

    scores = {"acc": acc, "miou": mean_IoU, "mf1": mean_f1}
    scores.update(cls_recall)
    scores.update(cls_precision)
    scores.update(cls_f1)
    scores.update(cls_iou)
    return scores
