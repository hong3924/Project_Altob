'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-03 06:17:57
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-05-19 13:34:49
FilePath: /user/mnt/HDD1/jinghung/YOLOX_deepsort_tracker/gmm/GMM1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np

# Video Path
#file_name = '../data/rec_2022_03_08_16_57_19.mp4'
#file_name = '../data/rec_2022_03_31_11_52_38.mp4'
#file_name = '../data/rec_2022_03_31_11_54_18.mp4'
#file_name = '../data/test_0310_103.mp4'
file_name = '../data/02.avi'

# Load Video and VideoWritter
cap = cv2.VideoCapture(file_name)
width = int(cap.get(3))
high = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
bg_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'background' + '.mp4', fourcc, 20, (width, high))
#fg_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'foreground' + '.mp4', fourcc, 20, (width, high))
#fgmask_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'fgmask' + '.mp4', fourcc, 20, (width, high), 0)
#new_fgmask_out = cv2.VideoWriter(file_name.split('/')[-1].split('.')[0] + '_' + 'new_fgmask' + '.mp4', fourcc, 20, (width, high), 0)

# MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)# varThreshold=1000

'''
# Get backgroung by mog2
for i in range(100):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    background = fgbg.getBackgroundImage()
    print(i)
    print(background.shape)
    #cv2.imwrite('bg'+str(i)+'.png', background)
    out.write(background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

'''
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    # Run MOG2 and get background
    fgmask = fgbg.apply(frame)
    
    background = fgbg.getBackgroundImage()
    print(background.shape)
    #cv2.imwrite('bg'+str(i)+'.png', background)
    out.write(background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    foreground=cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=fgmask)
    cv2.imwrite('fg.png', background)
'''

'''
for i in range(100):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    foreground=cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=fgmask)
    cv2.imwrite('fg.png', foreground)
    cv2.imwrite('fgmask.png', fgmask)
    
'''


def posprocess(img):
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #    cv2.THRESH_BINARY,11,2)
    #img = cv2.dilate(img, None, iterations=5)
    #img = cv2.erode(img, None, iterations=6)
    #img = cv2.dilate(img, (2,2), iterations=1)
    img = cv2.medianBlur(img, 3)
    #img = cv2.blur(img,(2,2))
    #ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #img = cv2.erode(img, (1,1), iterations=1)
    img = cv2.dilate(img, None, iterations=42)
    img = cv2.erode(img, None, iterations=42)
    img = cv2.dilate(img, None, iterations=8)
    return img


def findContours(img):
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def fill_area(noise, foreground, fgmask):
    # Fill noise will black
    cv2.drawContours(fgmask, noise, -1, (0,0,0), thickness=-1)
    # Fill foreground obj will white
    cv2.drawContours(fgmask, foreground, -1, (255,255,255), thickness=-1)
    return fgmask

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    #circle = np.zeros(frame.shape[0:2], dtype="uint8")
    #cv2.circle(circle, (520, 455), 140, 255, -1)
    #cv2.imwrite('circle.png', circle)
    fgmask = fgbg.apply(frame)
    '''
    new_fgmask = posprocess(fgmask)

    cnts = findContours(new_fgmask)
    minArea = 40000
    c_noise = []
    c_foreground = []
    for c in cnts:
        # if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < minArea:
            c_noise.append(c)
            #continue
        elif cv2.contourArea(c) >= minArea:
            c_foreground.append(c)
            #print(c_foreground)
        # draw an rectangle "around" the object
        #(x, y, w, h) = cv2.boundingRect(c)
        
        #cv2.rectangle(foreground, (x, y), (x + w, y + h), (0,0,255), 2)
    
    new_fgmask = fill_area(c_noise, c_foreground, new_fgmask)

    foreground = cv2.bitwise_and(frame, frame, mask=new_fgmask)
    '''
    background = fgbg.getBackgroundImage()
    
    # Store to videos
    #fgmask_out.write(fgmask)
    #new_fgmask_out.write(new_fgmask)
    #fg_out.write(foreground)
    bg_out.write(background)

cap.release()
cv2.destroyAllWindows()