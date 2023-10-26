'''
Author: your name
Date: 2022-02-24 07:45:17
LastEditTime: 2022-05-05 10:13:14
LastEditors: pha123661 pha123661@gmail.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /jinghung/YOLOX_deepsort_tracker/demo.py
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



def track_cap(model, file, output_path):
    
    # get model name
    m_temp = model.split('/')[-1].split('.')[0].split('_')
    model_type = m_temp[0] + '-' + m_temp[1]
    #print('model_type: ', model_type)

    # Initiate Tracker
    tracker = Tracker(model=model_type, ckpt=model, filter_class=['truck','car'])

    fr = 1

    # Load Video and VideoWritter
    cap = cv2.VideoCapture(file)  
    width  = int(500*cap.get(3)/cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path+file.split('/')[-1].split('.')[0]+'_'+model.split('/')[-1].split('.')[0]+'_'+str(fr)+'fr'+'.mp4', fourcc, 20, (width, 500))
    #print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    a = 0

    
    while True:
    #while a<=200:    
        _, im = cap.read()
        #print(im.shape)
        if im is None:
            break

        a += 1
        if a%fr != 0:
            continue
        
        
        print('frame:', a)  
        '''
        path = './results_test/output.txt'
        with open(path, 'a') as f:
            f.write('frme: ')
            f.write(str(a))
            f.write('\n')
        '''
        
        im = imutils.resize(im, height=500)
        #print(im.shape)
        image,_ = tracker.update(im, a)
        
        out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # cv2.imshow('demo', image)
        # cv2.waitKey(1)
        # if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
        #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('--model', type=str, default='./model/yolox_s.pth', help='yolox model')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='./results1/', help='output folder')
    args = parser.parse_args()

    if os.path.isfile(args.source):
        track_cap(args.model, args.source, args.output)
    else:
        track_images(args.path)
        