'''
Author: your name
Date: 2022-02-24 07:45:17
LastEditTime: 2022-07-02 14:11:36
LastEditors: pha123661 pha123661@gmail.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /jinghung/YOLOX_deepsort_tracker/tracker.py
'''
import sys
import numpy as np
sys.path.insert(0, './YOLOX')
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from utils.visualize import vis_track
from deep_sort.deep_sort.deep.feature_extractor import Extractor
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from queue import Queue
import os, time

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class_names = COCO_CLASSES

class Tracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='./model/yolox_s.pth', ):
        self.detector = Detector(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        self.filter_class = filter_class

        self.extractor = Extractor(cfg.DEEPSORT.REID_CKPT, use_cuda=True)

    def update(self, bg_im, fg_im, frame):
        # tmp = time.time()
        if False:
            # 0.06 ~ 0.07
            infos = self.detector.detect_imgs([bg_im, fg_im], visual=False)
            bg_info = infos[0]
            fg_info = infos[1]
        else:
            bg_info = self.detector.detect(bg_im, visual=False)
            fg_info = self.detector.detect(fg_im, visual=False)

            # print(f"bg_box: {bg_info['box_nums']}")
            # print(f"fg_box: {fg_info['box_nums']}")
        # print('[Time] Model Inference time', time.time() - tmp)

        bg_outputs = []
        fg_outputs = []

        # tmp = time.time()
        if fg_info['box_nums'] > 0:
            fg_bbox_xywh = []
            fg_scores = []

            for (x1, y1, x2, y2), class_id, score  in zip(fg_info['boxes'],fg_info['class_ids'],fg_info['scores']):
                if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                    continue
                fg_bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                fg_scores.append(score)
       
            fg_bbox_xywh = torch.Tensor(fg_bbox_xywh)
        # loop_1st =  time.time() - tmp 
            
        # tmp = time.time()
        if bg_info['box_nums'] > 0:
            bg_bbox_xywh = []
            bg_scores = []
     
            for (x1, y1, x2, y2), class_id, score  in zip(bg_info['boxes'],bg_info['class_ids'],bg_info['scores']):
                if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                    continue
                bg_bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                bg_scores.append(score)

            bg_bbox_xywh = torch.Tensor(bg_bbox_xywh)
        # loop_2st =  time.time() - tmp 

        
        ## ADD CODE ##
        ######################################################
        ## Using multiTehread to seperate FG/BG
        queue = Queue()
        threads = []
        threads.append(Thread(target=self.deepsort.update, args=([bg_bbox_xywh, bg_scores, bg_im, queue, frame, 0]), daemon=True))
        threads.append(Thread(target=self.deepsort.update, args=([fg_bbox_xywh, fg_scores, fg_im, queue, frame, 1]), daemon=True))
        threads[0].start()
        threads[1].start()

        for process in threads:
            process.join()


        ## Get the results of deepsort
        results = []
        for _ in range(2):
            results.append(queue.get())
        bg_outputs = results[0]
        fg_outputs = results[1]

        '''
        ## Determine the FG/BG objects to plot on the video

        # determine the obj in original video is fg/bg obj
        bg_obj = []
        fg_obj = []
        for indice, fg_bb in enumerate(fg_outputs):
            for bg_bb in bg_outputs:
                fg_bb1 = [fg_bb[0], fg_bb[1]] # origin obj bbox top-left (x1, x2)
                fg_bb2 = [fg_bb[2], fg_bb[3]] # origin obj bbox bottom-right (x2, y2)
                bg_bb1 = [bg_bb[0], bg_bb[1]] # background obj bbox top-left (x1, x2)
                bg_bb2 = [bg_bb[2], bg_bb[3]] # background obj bbox bottom-right (x2, y2)
                
                # determine the coordinates of the intersection rectangle
                x_left = max(fg_bb1[0], bg_bb1[0])
                y_top = max(fg_bb1[1], bg_bb1[1])
                x_right = min(fg_bb2[0], bg_bb2[0])
                y_bottom = min(fg_bb2[1], bg_bb2[1])
                
                if x_right < x_left or y_bottom < y_top:
                    overlap =  0.0
                
                else:
                    # calculate intersection
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)

                    # calculate overlap of fg_bbox_area and intersection_area
                    bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                    fg_bb_area = (fg_bb2[0] - fg_bb1[0]) * (fg_bb2[1] - fg_bb1[1])
                    overlap = intersection_area / float(fg_bb_area)
               
                    if overlap > 0.9:
                        bg_obj.append(indice)
         
            for i in range(len(fg_outputs)):
                if i not in bg_obj:
                    fg_obj.append(fg_outputs[i])
        '''
        
        '''
        # Generate bbox label
        root = ensure_path("/mnt/HDD1/jinghung/YOLOX_deepsort_tracker_mulThread/results_fuse_0615/bbox_results_04")
        filename = os.path.join(root, f"{frame}.txt")
        with open(filename, "w") as f:
            for track in bg_outputs:
                f.write("{} {} {} {} {}\n".format(int(track[4]), int(track[0]), int(track[1]), int(track[2]), int(track[3])))

            for track in fg_outputs:
                f.write("{} {} {} {} {}\n".format(int(track[4]), int(track[0]), int(track[1]), int(track[2]), int(track[3])))
        '''    


        # Plot the bbox on video
        image = vis_track(fg_im, fg_outputs, frame)
        image = vis_track(image, bg_outputs, frame)

        return image
        ######################################################
        

    def merge_output(self, bg_outputs, fg_outputs, bg_im, fg_im):
        # Remove fg feature in 'fg_features' if its feature is similar to bg features in 'bg_features'
        bg_features = self.get_features(bg_outputs, bg_im)
        fg_features = self.get_features(fg_outputs, fg_im)
        #print('fg_in\n', fg_outputs)
        #print('bg\n', bg_outputs)
        
        if (len(bg_features) != 0) and (len(fg_features) != 0):
            indexes = []
            for i in range(len(fg_features)):
                for j in range(len(bg_features)):
                    arr_vec_1 = [fg_features[i]]
                    arr_vec_2 = [bg_features[j]]
                    similarity = cosine_similarity(arr_vec_1, arr_vec_2)
                    print(f'i: {i}, j: {j}, sim{similarity}')
                    if similarity[0][0] >= 0.9:
                        indexes.append(i)
                        break

            #print('indexes\n', indexes)
            fg_outputs = np.delete(fg_outputs, indexes, axis = 0)
            #print('fg_out\n', fg_outputs)

        return bg_outputs, fg_outputs
    

    def get_features(self, outputs, ori_img):
        im_crops = []
        for output in outputs:
            x1,y1,x2,y2 = output[:4]
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
