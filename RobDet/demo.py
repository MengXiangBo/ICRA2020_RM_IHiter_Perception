import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from data import BaseTransform, RM_CLASSES
from data import config
import numpy as np
import cv2
import tools
import time
from decimal import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='RM Detection')

parser.add_argument('-v', '--version', default='slim_yolo_v2',
                    help='slim_yolo_v2, tiny_yolo_v3.')
parser.add_argument('--trained_model', default='weights/rm/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--mode', default='image',
                    type=str, help='Use the data from image, video or camera')
parser.add_argument('--path_to_img', default='data/image/0/',
                    type=str, help='The path to image files')
parser.add_argument('--path_to_vid', default='data/video/top.avi',
                    type=str, help='The path to video files')
parser.add_argument('--path_to_saveVid', default='data/video/result.avi',
                    type=str, help='The path to save the detection results video')
parser.add_argument('--visual_threshold', default=0.2, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')
args = parser.parse_args()


def vis(img, bbox_pred, scores, cls_inds, thresh):
    class_color = tools.CLASS_COLOR
    for i, box in enumerate(bbox_pred):
        if scores[i] > thresh:
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
            mess = '%s: %.3f' % (RM_CLASSES[int(cls_indx)], scores[i])
            cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img


def detect(net, device, transform, thresh, mode='image', path_to_img=None, path_to_vid=None, path_to_save=None):
    # dump predictions and assoc. ground truth to text file for now
    # I need to do :
    # opencv code to catch images in the real environment
    # My code here ....
    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            cv2.imshow('current frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            detections = net(x)      # forward pass
            torch.cuda.synchronize()
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            scale = np.array([[frame.shape[1], frame.shape[0],
                                frame.shape[1], frame.shape[0]]])
            bbox_pred, scores, cls_inds = detections
            # map the boxes to origin image scale
            bbox_pred *= scale

            frame_processed = vis(frame, bbox_pred, scores, cls_inds, thresh=thresh)
            cv2.imshow('detection result', frame_processed)
            cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        count = 0
        for file in os.listdir(path_to_img):
            img = cv2.imread(path_to_img + '/' + file, cv2.IMREAD_COLOR)
            x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            detections = net(x)      # forward pass
            torch.cuda.synchronize()
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            scale = np.array([[img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]]])
            bbox_pred, scores, cls_inds = detections
            # map the boxes to origin image scale
            bbox_pred *= scale

            img_processed = vis(img, bbox_pred, scores, cls_inds, thresh=thresh)
            cv2.imwrite('det_results/' + str(count)+'.png', img_processed)
            count += 1
            cv2.imshow('detection result', img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        os.makedirs('results', exist_ok=True)
        out = cv2.VideoWriter('results/output000.avi',fourcc, 40.0, (1280,720))
        i = 0
        while(True):
            ret, frame = video.read()
            
            if ret:
                # If you want to save each frame from the video, please use the following codes.
                # ------------------------- Save each frame ---------------------------
                # if i % 20 == 0:
                #     cv2.imwrite('data/image/robots_unprocessed_2/'+str(i // 20 + 2271)+'.png', frame)
                # i += 1
                # continue
                
                # ------------------------- Detection ---------------------------
                t0 = time.time()
                x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)

                torch.cuda.synchronize()
                t0 = time.time()
                detections = net(x)      # forward pass
                torch.cuda.synchronize()
                t1 = time.time()
                print("detection time used ", t1-t0, "s")
                # scale each detection back up to the image
                scale = np.array([[frame.shape[1], frame.shape[0],
                                    frame.shape[1], frame.shape[0]]])
                bbox_pred, scores, cls_inds = detections
                # map the boxes to origin image scale
                bbox_pred *= scale
                
                frame_processed = vis(frame, bbox_pred, scores, cls_inds, thresh=thresh)
                out.write(frame_processed)
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_classes = len(RM_CLASSES)
    cfg = config.rm_ab

    if args.version == 'slim_yolo_v2':
        from models.slim_yolo_v2 import YOLOv2slim
        net = YOLOv2slim(device, input_size=cfg['min_dim'], num_classes=num_classes, trainable=False, anchor_size=config.ANCHOR_SIZE)
        print('Let us test slim_yolo_v2 on the RM dataset ......')

    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        net = YOLOv3tiny(device, input_size=cfg['min_dim'], num_classes=num_classes, trainable=False, anchor_size=config.MULTI_ANCHOR_SIZE)
        print('Let us test tiny-yolo-v3 on the VOC0712 dataset ......')


    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')
    
    # run
    if args.mode == 'camera':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)), 
                    thresh=args.visual_threshold, mode=args.mode)
    elif args.mode == 'image':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)), 
                    thresh=args.visual_threshold, mode=args.mode, path_to_img=args.path_to_img)
    elif args.mode == 'video':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                    thresh=args.visual_threshold, mode=args.mode, path_to_vid=args.path_to_vid, path_to_save=args.path_to_saveVid)


if __name__ == '__main__':
    run()