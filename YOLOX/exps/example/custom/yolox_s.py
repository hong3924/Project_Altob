'''
Author: pha123661 pha123661@gmail.com
Date: 2022-06-19 07:32:36
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-20 08:27:57
FilePath: /jinghung/YOLOX/exps/example/custom/yolox_s.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/NYCU_CAM"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 71

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
