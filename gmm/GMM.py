'''
Author: pha123661 pha123661@gmail.com
Date: 2022-05-03 06:17:57
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-09 10:44:51
FilePath: /user/mnt/HDD1/jinghung/YOLOX_deepsort_tracker/gmm/GMM1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GMM Background Extration")
    parser.add_argument('--video', type=str, default='../originalvideo', help='original video path')
    parser.add_argument('--output', type=str, default='../', help='output video path')
    args = parser.parse_args()

    # Load Video and VideoWritter
    filepath = args.video 
    outpath = args.output
    cap = cv2.VideoCapture(filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    high = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    bg_out = cv2.VideoWriter(outpath + filepath.split('/')[-1].split('.')[0] + '_' + 'background' + '.mp4', fourcc, fps, (width, high))

    # GMM initialize (MOG2)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)# varThreshold=1000


    progress = tqdm(total = length)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        # Run MOG2, get foreground mask and background image
        fgmask = fgbg.apply(frame)
        background = fgbg.getBackgroundImage()
        
        # Write the background image into output video
        bg_out.write(background)
        
        progress.update(1)

    cap.release()
    progress.close()