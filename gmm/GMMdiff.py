'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-03 06:17:57
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-05-26 10:26:47
FilePath: /user/mnt/HDD1/jinghung/YOLOX_deepsort_tracker/gmm/GMM1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np

# Video Path
file_name = '../data/rec_2022_03_08_16_57_19.mp4'
#file_name = '../data/rec_2022_03_31_11_52_38.mp4'
#file_name = '../data/rec_2022_03_31_11_54_18.mp4'
#file_name = '../data/test_0310_103.mp4'
#file_name = '../data/02.avi'

# Load Video and VideoWritter
cap = cv2.VideoCapture(file_name)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
high = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#bg_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'background' + '.mp4', fourcc, 20, (width, high))
#fg_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'foreground' + '.mp4', fourcc, 20, (width, high))
#fgmask_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'fgmask' + '.mp4', fourcc, 20, (width, high), 0)
#new_fgmask_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'new_fgmask' + '.mp4', fourcc, 20, (width, high), 0)
fgmask_diff_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'fgmask_diff' + '.mp4', fourcc, fps, (width, high), 0)

# MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)# varThreshold=1000


def toGray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = fgbg.apply(frame)

    background = fgbg.getBackgroundImage()
    
    fgmask_diff = cv2.absdiff(toGray(frame), toGray(background))

    # Store to videos
    #fgmask_out.write(fgmask)
    #new_fgmask_out.write(new_fgmask)
    #fg_out.write(foreground)
    #bg_out.write(background)
    fgmask_diff_out.write(fgmask_diff)
    

cap.release()
cv2.destroyAllWindows()