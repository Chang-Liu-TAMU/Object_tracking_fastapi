# @Time: 2022/5/11 18:23
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:yolov5_detector.py

import logging
import os
from collections import namedtuple, OrderedDict
from PIL import ImageColor

import torch
import tensorrt as trt
import numpy as np
import cv2


from .utils import scale_coords, time_sync, clip_coords, timer_wrapper
from . import tools
from config.label import get_label_name



DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)

logger = logging.getLogger(__name__)

class Yolov5Detector:
    def __init__(self, file_name):
        self.mode = None
        self.gpu_mode = 0

        self.frame = None
        self.detector = None
        self.input_size = 640
        self.nms_threshold = 0.45
        self.skip_scores = None
        self.server_mode = None
        self.num_classes = None
        self.skip_display = None
        self.visualization = None
        self.detect_threshold = 0.25
        self.model_file = file_name
        self.classes = [0]
        self._load_model()


    def _run_inference_local(self, image_np):
        image_data = image_np
        # image_data = self._yolov5_preporcess(np.copy(image_np), [self.input_size, self.input_size])
        y = None
        device = torch.device("cuda:0")
        # img = image_data
        img = tools.letterbox(image_data, new_shape=self.input_size)[0]
        # cv2.imwrite("letterbox.jpg", img)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        # img = img.float()
        img /= 255 # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # im = torch.zeros(*imgsz, dtype=torch.float, d
        assert img.shape == self.bindings['images'].shape, (img.shape, self.bindings['images'].shape)
        t1 = time_sync()
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        pred_bbox = self.bindings['output'].data
        t2 = time_sync()
        text = "running time: {} seconds".format(str(t2 - t1))
        # print(text)
        if isinstance(pred_bbox, np.ndarray):
            device = torch.device("cuda:0")
            pred_bbox = torch.tensor(pred_bbox, device=device)
        # pred_bbox = torch.from_numpy(pred_bbox).to(device)
        # print(pred_bbox.shape)
        # print(pred_bbox[0][1])
        pred_bbox = tools.non_max_suppression(pred_bbox, self.detect_threshold, self.nms_threshold,
                                              classes=self.classes)[0]
        # print(pred_bbox)
        # pred_box: x, y, x, y, conf, cls
        pred_bbox[:, :4] = Yolov5Detector.__onnx_post_processing(img.shape[2:], pred_bbox[:, :4], image_np.shape).round()
        # dets_val = pred_bbox[:, [5, 4, 0, 1, 2, 3]]

        dets_val = pred_bbox.cpu().detach().numpy()[:, [5, 4, 0, 1, 2, 3]]

        # dets_val = pred_bbox.cpu().detach().numpy()
        # # print("detectoins: ", dets_val)
        # detections = []
        # for i in range(len(dets_val)):
        #     tlbr = dets_val[i, :4]
        #     # tlbr = xywh_2_tlbr(nms_dets[i, :4])
        #     label = int(dets_val[i, 5])
        #     conf = dets_val[i, 4]
        #     # if 0 < area(tlbr) <= max_area and aspect_ratio(tlbr) >= min_ar:
        #     detections.append((tlbr, label, conf))
        # # print(detections)
        # detections = np.fromiter(detections, DET_DTYPE, len(detections)).view(np.recarray)

        # return detections
        return dets_val

    @staticmethod
    def __onnx_post_processing(img0, boxes, img):
        return scale_coords(img0, boxes, img)

    @staticmethod
    def _yolov5_preporcess(image, target_size, gt_boxes=None):
        # image = image.astype(np.float32)

        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=0)  # 128.0

        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes


    @timer_wrapper
    def _warm_up(self):
        im = (1, 3, self.input_size, self.input_size)
        device = torch.device("cuda")
        im = torch.zeros(*im, dtype=torch.float, device=device)  # input
        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        _ = self.bindings['output'].data
        logger.info("Successfully warmed up tensorrt engine. ")


    def _load_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("we are testing ====>>>>", self.model_file)

        self.bindings, self.binding_addrs, self.context = self.load_tensorrt_model(self.model_file)
        self._warm_up()


    def load_tensorrt_model(self, model_file):
        logger.info(f'Loading {self.model_file} for TensorRT inference...')
        # import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        # check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        trt_logger = trt.Logger(trt.Logger.INFO)
        with open(self.model_file, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        fp16 = False  # default updated below
        device = torch.device("cuda:0")
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if model.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        batch_size = bindings['images'].shape[0]
        return bindings, binding_addrs, context

    def __call__(self, image_np):
        return self._run_inference_local(image_np)

    def draw(self, detection, img, color="RED",):
        color = ImageColor.getcolor(color, "RGB")[::-1]
        try:
            for i in range(len(detection)):
                x1, y1, x2, y2, conf, label = detection[i]
                # if int(c) == classification:
                cv2.rectangle(img=img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color,
                              thickness=3)
                logo = f"{get_label_name(int(label))} {conf:.3f}"
                cv2.putText(img, logo, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 69, 255), 1,
                            cv2.LINE_AA)
        except Exception as e:
            logger.warning("drawing failed")
            print(e)
            return
        # cv2.imwrite("demo.jpg", img)
        # cv2.imwrite("./moder_family_detect.jpg", img)
