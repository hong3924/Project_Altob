'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-26 05:12:14
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-07-02 10:36:25
FilePath: /jinghung/YOLOX_deepsort_tracker_mulThread/detector.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# import sys
# sys.path.insert(0, './YOLOX')
import torch
import numpy as np
import cv2


import sys
sys.path.append(r"/mnt/HDD1/jinghung/YOLOX_deepsort_tracker_mulThread/YOLOX")

from YOLOX.yolox.data.data_augment import preproc
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp.build import get_exp_by_name,get_exp_by_file
from YOLOX.yolox.utils import postprocess
from utils.visualize import vis



COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)




class Detector():
    """ 图片检测器 """
    def __init__(self, model='yolox-x', ckpt='/mnt/HDD1/jinghung/YOLOX_deepsort_tracker_mulThread/model_fine_tune/yolox_x.pth'):
        super(Detector, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')
        # self.exp = get_exp_by_name(model)
        self.exp = get_exp_by_file("/mnt/HDD1/jinghung/YOLOX_deepsort_tracker_mulThread/YOLOX/exps/example/yolox_coco_x.py")
        self.test_size = self.exp.test_size  # TODO: 改成图片自适应大小
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])



    def detect(self, raw_img, visual=True, conf=0.5):
        """
        raw_img: (w, h, 3)
        """
        info = {}
        img, ratio = preproc(raw_img, self.test_size)#, COCO_MEAN, COCO_STD
        info['raw_img'] = raw_img
        info['img'] = img

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre  # TODO:用户可更改
            )[0].cpu().numpy()
        
        if outputs[0] is None:
            info['boxes'], info['scores'], info['class_ids'],info['box_nums']=None,None,None,0
        else:
            info['boxes'] = outputs[:, 0:4]/ratio
            info['scores'] = outputs[:, 4] * outputs[:, 5]
            info['class_ids'] = outputs[:, 6]
            info['box_nums'] = outputs.shape[0]
        # 可视化绘图
        if visual:
            info['visual'] = vis(info['raw_img'], info['boxes'], info['scores'], info['class_ids'], conf, COCO_CLASSES)
        return info

    def detect_imgs(self, raw_imgs, visual=True, conf=0.5):
        """
        raw_img: list of images [(w, h, 3), (w, h, 3)......]
        """
        infos = [{}, {}]
        img1, ratio = preproc(raw_imgs[0], self.test_size)#, COCO_MEAN, COCO_STD
        img2, ratio = preproc(raw_imgs[1], self.test_size)#, COCO_MEAN, COCO_STD

        # (2, h, w, 3)
        img = np.stack([img1, img2], axis=0)

        img = torch.from_numpy(img)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre  # TODO:用户可更改
            )
            output1 = outputs[0].cpu().numpy()
            output2 = outputs[1].cpu().numpy()


        infos[0]['boxes'] = output1[:, 0:4]/ratio
        infos[0]['scores'] = output1[:, 4] * output1[:, 5]
        infos[0]['class_ids'] = output1[:, 6]
        infos[0]['box_nums'] = output1.shape[0]

        infos[1]['boxes'] = output2[:, 0:4]/ratio
        infos[1]['scores'] = output2[:, 4] * output2[:, 5]
        infos[1]['class_ids'] = output2[:, 6]
        infos[1]['box_nums'] = output2.shape[0]
            
        return infos

    def bbox_iou_broacast(bboxes1, bboxes2):
        """
        @param bboxes1: (a, b, ..., 4)
        @param bboxes2: (A, B, ..., 4)
            x:X is 1:n or n:n or n:1
        @return (max(a,A), max(b,B), ...)
        ex) (4,):(3,4) -> (3,)
            (2,1,4):(2,3,4) -> (2,3)
        expand_dims
        concat
        divide_no_nan
        """
        # (fg, )
        bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
        bboxes1_area = tf.expand_dims(bboxes1_area, axis=-1) # (fg, 1)
        # (bg, )
        bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
        bboxes1_area = tf.expand_dims(bboxes1_area, axis=0) # (1, bg)

        # (fg, 4)  => (fg, 1, 4)
        bboxes1_coor = tf.concat(  # x1y1x2y2
            [
                bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
                bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes1_coor = tf.expand_dims(bboxes1_coor, axis=1)

        # (bg, 4)  => (1, bg, 4)
        bboxes2_coor = tf.concat(
            [
                bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
                bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes2_coor = tf.expand_dims(bboxes2_coor, axis=0)

        # (fg, bg, 2)
        left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        # (fg, bg, 2)
        right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

        # (fg, bg, 2)
        inter_section = tf.maximum(right_down - left_up, 0.0)
        # (fg, bg)
        inter_area = inter_section[..., 0] * inter_section[..., 1]


        # (fg, bg) = (fg, 1) + (1, bg) - (fg, bg)
        union_area = bboxes1_area + bboxes2_area - inter_area

        iou = tf.math.divide_no_nan(inter_area, union_area)

        # (fg, bg)
        return iou


if __name__=='__main__':
    detector = Detector()
    img = cv2.imread('./1000.jpg')
    out = detector.detect(img)
    print(out)
