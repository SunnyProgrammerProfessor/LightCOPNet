# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：logger_record.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/3/21 23:35 
"""
import sys
import time


class Logger(object):
    def __init__(self, logger_path):
        super(Logger, self).__init__()
        self.logger_path = logger_path
        self.terminal = sys.stdout  # 就相当于print()
        now = time.strftime("%c")  # 获取本地相应的日期表示和时间表示
        self.write(f"==============={now}===============")
        self.write("\n")

    def write(self, message):
        self.terminal.write(message)
        with open(self.logger_path, mode="a") as f:
            f.write(message)

    def write_dict(self, dict):
        message = ""
        for k, v in dict.items():
            message += f"{k}: {v:.7f} "
        self.write(message)

    def write_dict_str(self, dict):
        message = ""
        for k, v in dict.items():
            message += f"{k}: {v} "
        self.write(message)

    def flush(self):
        self.terminal.flush()


class Timer:
    def __init__(self, starting_msg=None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    # 更新进度估计
    def update_progress(self, progress):  # progress:进度
        self.elapsed = time.time() - self.start  # 计算已用时间
        self.est_total = self.elapsed / progress  # 估计总计时间，progress代表着进度
        self.est_remaining = self.est_total - self.elapsed  # 估计剩余时间
        self.est_finish = int(self.start + self.est_total)  # 估计结束时间

    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))  # 以系统本地日期返回估计结束时间

    def str_estimated_remaining(self):
        return str(self.est_remaining / 3600) + 'h'  # 估计剩余时间还有多少小时

    def estimated_remaining(self):
        return self.est_remaining / 3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start  # 返回上一阶段到目前经过的时间

    def reset_stage(self):
        self.stage_start = time.time()  # 重新设置阶段计时器

    def lapse(self):  # 整合get_stage_elapsed()和reset_stage
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out
