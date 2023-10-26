'''
Author: pha123661 pha123661@gmail.com
Date: 2022-02-24 08:30:06
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-14 10:17:29
FilePath: /YOLOX_deepsort_tracker/demo1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from collections import deque
from common import clock, draw_str, StatValue

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

    # if os.path.isfile(args.fg_source) and os.path.isfile(args.fg_source):
    #     track_cap(args.model, args.bg_source, args.fg_source, args.output)

    fr = 1
    bg_cap = cv2.VideoCapture(args.bg_source)  
    ori_cap = cv2.VideoCapture(args.fg_source)  
    fps = ori_cap.get(cv2.CAP_PROP_FPS)
    length = int(ori_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(500*ori_cap.get(3)/ori_cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output+args.fg_source.split('/')[-1].split('.')[0]+'_'+args.model.split('/')[-1].split('.')[0]+'_'+str(fr)+'fr'+'.mp4', fourcc, fps, (width, 500))

    def process_frame(frame, t0):
        # some intensive computation...
        # frame = cv.medianBlur(frame, 19)
        # frame = cv.medianBlur(frame, 19)
        # return frame, t0

        bg_im = imutils.resize(bg_im, height=500)
        ori_im = imutils.resize(ori_im, height=500)
        image = tracker.update(bg_im, ori_im, a)
        out.write(image)

    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    threaded_mode = True

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value*1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
            cv2.imshow('threaded video', res)
        if len(pending) < threadn:
            _ret, frame = cap.read()
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), t))
            # else:
            #     task = DummyTask(process_frame(frame, t))
            pending.append(task)

        ch = cv2.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break

    print('Done')
