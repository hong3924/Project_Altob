'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-03 06:17:57
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-27 08:11:33
FilePath: /user/mnt/HDD1/jinghung/YOLOX_deepsort_tracker/gmm/GMM1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import argparse
import numpy as np


# A list of the paths of your videos
# path = './../data/Croped_NYCU_parking_camera1_04.mp4'
path = './../data/test_Trim.mp4'
# Create a new video
cap = cv2.VideoCapture(path)


a = 0
while True:
    a+=1
    ret, frame = cap.read()
    if frame is None:
        break
    output_folder = './test_Trim_imgs/'
    output_name = output_folder +  str(a) + '.jpg'
    cv2.imwrite(output_name, frame)


cap.release()