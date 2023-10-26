'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-03 07:19:20
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-05-03 12:29:20
FilePath: /user/mnt/HDD1/jinghung/YOLOX_deepsort_tracker/gmm/mog2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import ctypes as C
import cv2

libmog = C.cdll.LoadLibrary("./libmog2.so")

def getfg(img):
    (rows, cols) = (img.shape[0], img.shape[1])
    res = np.zeros(dtype=np.uint8, shape=(rows, cols))
    libmog.getfg(img.shape[0], img.shape[1],
                       img.ctypes.data_as(C.POINTER(C.c_ubyte)),
                       res.ctypes.data_as(C.POINTER(C.c_ubyte)))
    return res


def getbg(img):
    (rows, cols) = (img.shape[0], img.shape[1])
    res = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))

    libmog.getbg(rows, cols, res.ctypes.data_as(C.POINTER(C.c_ubyte)))
    return res


if __name__ == '__main__':
    c = cv2.VideoCapture('../data/rec_2022_03_08_16_57_19.mp4')
    '''
    while 1:
        
        _, f = c.read()
        cv2.imshow('f', f)
        cv2.imshow('fg', getfg(f))
        cv2.imshow('bg', getbg(f))
        if cv2.waitKey(1) == 27:
            exit(0)
    '''

    for i in range(100):
        _, f = c.read()
        cv2.imwrite('bg.png', getbg(f))
