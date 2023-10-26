'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-03 06:51:11
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-05-16 05:58:18
FilePath: /YOLOX_deepsort_tracker/gmm/GMM2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np

# Video Path
file_name = '../data/rec_2022_03_08_16_57_19.mp4'
#file_name = '../data/rec_2022_03_31_11_52_38.mp4'
#file_name = '../data/rec_2022_03_31_11_54_18.mp4'
#file_name = '../data/test_0310_103.mp4'

# Load Video and VideoWritter
cap = cv2.VideoCapture(file_name)
width = int(cap.get(3))
high = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Output directions 
bg_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'background_2' + '.mp4', fourcc, 20, (width, high))
fg_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'foreground_2' + '.mp4', fourcc, 20, (width, high))
fgmask_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'fgmask_2' + '.mp4', fourcc, 20, (width, high), 0)
new_fgmask_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'new_fgmask_2' + '.mp4', fourcc, 20, (width, high), 0)

# MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)# varThreshold=1000

# Posprocess for fgmask
def posprocess(img):
    img = cv2.medianBlur(img, 3)
    img = cv2.dilate(img, None, iterations=42)
    img = cv2.erode(img, None, iterations=42)
    #img = cv2.dilate(img, None, iterations=8)
    return img

# Finding Contours of fgmask
def findContours(img):
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts

# Fill the Contours in fgmask
def fill_area(noise, fgobj, fgmask):
    # Fill noise will black
    cv2.drawContours(fgmask, noise, -1, (0,0,0), thickness=-1)

    # Fill foreground obj will white
    cv2.drawContours(fgmask, fgobj, -1, (255,255,255), thickness=-1)

    for c in fgobj:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(fgmask, (x, y), (x + w, y + h), (255,255,255), thickness=-1)

    return fgmask

# Main function
while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = fgbg.apply(frame)
    new_fgmask = posprocess(fgmask)

    cnts = findContours(new_fgmask)
    minArea = 40000
    c_noise = []
    c_fgobj = []
    for c in cnts:
        # if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < minArea:
            c_noise.append(c)
            #continue
        elif cv2.contourArea(c) >= minArea:
            c_fgobj.append(c)
            #print(c_foreground)
        # draw an rectangle "around" the object
        #(x, y, w, h) = cv2.boundingRect(c)
        
        #cv2.rectangle(foreground, (x, y), (x + w, y + h), (0,0,255), 2)
    
    new_fgmask = fill_area(c_noise, c_fgobj, new_fgmask)

    foreground = cv2.bitwise_and(frame, frame, mask=new_fgmask)
    background = fgbg.getBackgroundImage()
    
    # Store to videos
    fgmask_out.write(fgmask)
    new_fgmask_out.write(new_fgmask)
    fg_out.write(foreground)
    bg_out.write(background)

cap.release()
cv2.destroyAllWindows()