'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-05 11:46:39
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-05-13 12:41:06
FilePath: /user/mnt/HDD1/jinghung/YOLOX_deepsort_tracker/gmm/transform.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np

# Video Path
#file_name = '../data/rec_2022_03_08_16_57_19.mp4'
file_name = '../data/rec_2022_03_31_11_52_38.mp4'
#file_name = '../data/rec_2022_03_31_11_54_18.mp4'
#file_name = '../data/test_0310_103.mp4'

# Load Video and VideoWritter
cap = cv2.VideoCapture(file_name)
width = int(cap.get(3))
high = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'transform' + '.mp4', fourcc, 20, (width, high))

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    out.write(frame)

cap.release()
cv2.destroyAllWindows()