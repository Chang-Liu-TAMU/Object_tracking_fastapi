# @Time: 2022/6/2 15:46
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:blueprint_.py

'''
deepsort = deepsort()
detector = yolov5()
modify mechansim of counting people going in and outside
v = video_stream
for frame in v:
    detections = detector.run(frame)
    tracks, i, o = deepsort.update(detections)
    we got info about tracks in each frame
    some_ui_functions(tracks)
    show_data(i, o)
    always_show_virtual_line(pre_defined)
end
'''
# from __future__ import absolute_import

from deep_sort.deep_sort import deep_sort
from object_detector.yolov5_detector import Yolov5Detector
import cv2
import os
import numpy as np
import tensorrt as trt
import platform
import cv2
from PIL import Image
from PIL import ImageColor
import os
import time

from utils.cv_utils import *

class Config:
    top_left = (275, 161)
    width = 108
    height = 120

boundary = (275, 161, 108, 120)


# drv.init()
# if platform.machine() == 'x86_64':
#     ctypes.CDLL('../object_detection//libmyplugins_x86_64.so')
# elif platform.machine() == 'aarch64':
#     ctypes.CDLL(os.path.join('libmyplugins_arm.so'))

def help(det, target):
    if det.shape != (0,):
        if det.ndim == 1:
            det = det.reshape((1, det.size))
        confidence = det[:, 1]
        cls = det[:, 0]
        needed = cls == target
        # print(set(cls))
        coords = det[:, 2:]
        coords[:, 2] -= coords[:, 0]
        coords[:, 3] -= coords[:, 1]
        return confidence[needed], coords[needed]
        # return confidence, coords
    else:
        return [], []

video_path = "./video.mp4"
# video_path = 0
cap = cv2.VideoCapture(video_path)
# model = yolo.YoloV5(model_file="../object_detection/yolov5_x86_64_1080ti.engine")
model = Yolov5Detector("../models/yolov5s6.engine")
tracker = deep_sort.DeepSort(model_path="../deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
if cap.isOpened():
    print("File opened successfully")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS of this video if {}".format(fps))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Video width * height is {} * {}".format(width, height))
    person_out = 0
    person_in = 0
    a = True
    frame_count = 0
    start = time.time()
    while cap.isOpened():
    # for i in range(1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            end = time.time()
            print(f"FPS: {frame_count / (end - start)}:.2f")
            break
        else:
            frame_count += 1

        # detection = model.inference(frame)
        # confidence, coords = help(detection, target=1)
        detection = model(frame)
        confidence, coords = help(detection, target=0)
        tracks, person_out, person_in = tracker.update(coords, confidence, frame, person_out, person_in, boundary, orientation="in_out")
        # print(tracks)
        # print(type(tracks))
        if isinstance(tracks, np.ndarray):
            # print(tracks.shape)
            # print("shape of tracks is {}".format(tracks.shape))
            # print(tracks)
            # print("id is : {}".format(tracks[:, 5]))
            # print(tracks)
            ids = tracks[:, 4]
            tracks = tracks[:, :4]
            # print(ids)
            # print(tracks)
            draw_tracks(tracks, frame, track_id=ids, p_out=person_out, p_in=person_in, config=Config)
            if a:
                cv2.imwrite("demo.jpg", frame)
                a = False
        cv2.imshow("hello", frame)
        key = cv2.waitKey(delay=1)
        if key == ord("q"):
            break
    cap.release()
    # assert isinstance(frame, np.ndarray), "type error"
    # model.pop()
else:
    print("fail to open file")