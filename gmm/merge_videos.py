'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-03 06:17:57
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-15 10:18:04
FilePath: /user/mnt/HDD1/jinghung/YOLOX_deepsort_tracker/gmm/GMM1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import argparse
import numpy as np


# A list of the paths of your videos
# path_1 = './../data/rec_2022_03_31_11_52_38.mp4'
# path_2 = './../data/rec_2022_03_31_11_54_18.mp4'
# path_1 = './../data/rec_2022_03_08_16_57_19.mp4'
# path_2 = './../data/rec_2022_03_08_16_58_16.mp4'
path_1 = './../data/03.mp4'

# Create a new video
cap_1 = cv2.VideoCapture(path_1)
# cap_2 = cv2.VideoCapture(path_2)
fps = cap_1.get(cv2.CAP_PROP_FPS)
width = int(cap_1.get(3))
high = int(cap_1.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# new_video = cv2.VideoWriter('./../data/03.mp4', fourcc, fps, (width, high))
new_video = cv2.VideoWriter('./../data/03_trim.mp4', fourcc, fps, (width, high))

a = 0
while True:
    a+=1

    ret, frame = cap_1.read()
    if frame is None:
        break

    if a%5 != 0:
        new_video.write(frame)
    

# while True:
#     ret, frame = cap_2.read()
#     if frame is None:
#         break

#     new_video.write(frame)

cap_1.release()
# cap_2.release()