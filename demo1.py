'''
Author: pha123661 pha123661@gmail.com
Date: 2022-02-24 08:30:06
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-14 14:40:44
FilePath: /YOLOX_deepsort_tracker/demo1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob
from tqdm import tqdm

# os.environ["cuda_visible_devices"] = '1'  #指定第一塊gpu

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


def track_cap(model, bg_file, fg_file, output_path):
    # get model name
    m_temp = model.split('/')[-1].split('.')[0].split('_')
    model_type = m_temp[0] + '-' + m_temp[1]


    # Initiate Tracker
    tracker = Tracker(model=model_type, ckpt=model, filter_class=['truck','car'])

    # Determine how many frames to process once
    fr = 1

    # Load Video and VideoWritter
    bg_cap = cv2.VideoCapture(bg_file)  
    ori_cap = cv2.VideoCapture(fg_file)  
    fps = ori_cap.get(cv2.CAP_PROP_FPS)
    length = int(ori_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(500*ori_cap.get(3)/ori_cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path+fg_file.split('/')[-1].split('.')[0]+'_'+model.split('/')[-1].split('.')[0]+'_'+str(fr)+'fr'+'.mp4', fourcc, fps, (width, 500))

    
    # Frame counter
    a = 0

    # Progress bar
    progress = tqdm(total = length)


    while True:
        _, bg_im = bg_cap.read()
        _, ori_im = ori_cap.read()

        if bg_im is None:
            break

        # Frame counter
        a += 1

        # Process every 'fr' frame
        if a%fr != 0:
            continue
        
        #print('frame:', a)  
        
        bg_im = imutils.resize(bg_im, height=500)
        ori_im = imutils.resize(ori_im, height=500)

        # Strat tracking from which frames
        if(a >= 0):
            image = tracker.update(bg_im, ori_im, a)

            out.write(image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        progress.update(1)

    bg_cap.release()
    ori_cap.release()
    out.release()
    cv2.destroyAllWindows()
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('--model', type=str, default='./model/yolox_x.pth', help='yolox model')
    parser.add_argument('--bg_source', type=str, default='0', help='background source')
    parser.add_argument('--fg_source', type=str, default='0', help='foreground source')
    parser.add_argument('--output', type=str, default='./results_fuse/', help='output folder')
    args = parser.parse_args()

    if os.path.isfile(args.fg_source) and os.path.isfile(args.fg_source):
        track_cap(args.model, args.bg_source, args.fg_source, args.output)
    #else:
    #    track_images(args.path)
        