'''
Author: pha123661 pha123661@gmail.com
Date: 2022-02-24 08:30:06
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-08 06:32:59
FilePath: /YOLOX_deepsort_tracker/demo1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob


def crop_cap(seconds, output_path, camera, number):

    # Load Video and VideoWritter
    if camera == '0':
        cap = cv2.VideoCapture('http://root:16660935@140.113.224.21/axis-cgi/mjpg/video.cgi?e_gamma=100&.mjpg')
        x = 0
        y = 300
        w = 1280
        h = 500

    if camera == '1':
        cap = cv2.VideoCapture('http://root:16660935@140.113.224.22/axis-cgi/mjpg/video.cgi?e_gamma=100&.mjpg')
        x = 50
        y = 290
        w = 1230
        h = 490

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f'fps: {fps}')
    # print(f'width: {cap.get(3)}')
    # print(f'height: {cap.get(4)}')
    width  = int(w)
    height = int(h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path+'Croped_NYCU_parking_camera'+camera+'_'+number+'.mp4', fourcc, fps, (width, height))

    # Frame counter
    a = 0

    while a < fps * seconds:

        a += 1

        _, im = cap.read()
        # Croping image
        crop_im = im[y:y+h, x:x+w]
        out.write(crop_im)
        # cv2.imwrite('./results_fuse/crop.jpg', crop_im)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('--seconds', type=int, default='300', help='How long the video you want?')
    parser.add_argument('--output', type=str, default='./results_fuse/', help='output folder')
    parser.add_argument('--camera', type=str, default='0', help='0 or 1')
    parser.add_argument('--number', type=str, default='00', help='the i-th videos')
    args = parser.parse_args()

    crop_cap(args.seconds, args.output, args.camera, args.number)