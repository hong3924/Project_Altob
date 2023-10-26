'''
Author: pha123661 pha123661@gmail.com
Date: 2022-02-24 08:30:06
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-07 07:33:52
FilePath: /YOLOX_deepsort_tracker/demo1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob


def track_images(img_dir):
    tracker = Tracker(model='yolox-s', ckpt='./yolox_s.pth',filter_class=['truck','person','car'])
    imgs = glob(os.path.join(img_dir,'*.png')) + glob(os.path.join(img_dir,'*.jpg')) + glob(os.path.join(img_dir,'*.jpeg'))
    for path in imgs:
        im = cv2.imread(path)
        im = imutils.resize(im, height=400)
        image,_ = tracker.update(im)
        #image = imutils.resize(image, height=500)

        cv2.imshow('demo', image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cv2.destroyAllWindows()


def track_cap(model, output_path, camera):

    # get model name
    m_temp = model.split('/')[-1].split('.')[0].split('_')
    model_type = m_temp[0] + '-' + m_temp[1]

    # Initiate Tracker
    tracker = Tracker(model=model_type, ckpt=model, filter_class=['truck','car'])

    # Determine how many frames to process once
    fr = 1

    # Load Video and VideoWritter
    if camera == '0':
        cap = cv2.VideoCapture('http://root:16660935@140.113.224.21/axis-cgi/mjpg/video.cgi?e_gamma=100&.mjpg')

    if camera == '1':
        cap = cv2.VideoCapture('http://root:16660935@140.113.224.22/axis-cgi/mjpg/video.cgi?e_gamma=100&.mjpg')
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'fps: {fps}')
    width  = int(500*cap.get(3)/cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path+'NYCU_parking_'+camera+'_4_'+model.split('/')[-1].split('.')[0]+'_'+str(fr)+'fr'+'.mp4', fourcc, fps, (width, 500))


    # MOG2
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)# varThreshold=1000


    # Frame counter
    a = 0

    while True:
        _, im = cap.read()
        
        # Frame counter
        a += 1

        # Process every 'fr' frame
        if a%fr != 0:
            continue
        
        print('frame:', a) 

        fgmask = fgbg.apply(im)
        background = fgbg.getBackgroundImage()

        bg_im = imutils.resize(background, height=500)
        fg_im = imutils.resize(im, height=500)

        # Start tracking
        image= tracker.update(bg_im, fg_im, a)

        # Write the output in video
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('--model', type=str, default='./model/yolox_x.pth', help='yolox model')
    parser.add_argument('--output', type=str, default='./results_fuse/', help='output folder')
    parser.add_argument('--camera', type=str, default='0', help='0 or 1')
    args = parser.parse_args()

    track_cap(args.model, args.output, args.camera)