import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3,
                 nms_max_overlap=1.0, max_iou_distance=0.7, max_age=200,
                 n_init=3, nn_budget=50, use_cuda=True, device=0):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda, device=device)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def judge_in_out(self, coor, boundary):
        x1, y1, x2, y2 = [int(x) for x in coor]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        top_x, top_y = boundary[0], boundary[1]
        if mid_x >= top_x and mid_x <= (mid_x + boundary[2]) and \
                mid_y >= top_y and mid_y <= (top_y + boundary[3]):
            return True
        else:
            return False


    def update(self, bbox_xywh, confidences, ori_img, personOut, personIn, boundary,
                   offset_ratio=0.5, orientation="horizontal", logger=None):
        self.height, self.width = ori_img.shape[:2]

        self.is_vertical = True if orientation == "vertical" else False

        # output bbox, identities etc
        if self.is_vertical:
            base_length = self.width
        else:
            base_length = self.height
        final_offset = int(base_length * offset_ratio)

        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = bbox_xywh
        # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections=detections, offset=final_offset, orientation=orientation, boundary=boundary)

        outputs = []
        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # update trace
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            # print("track id is : {}".format(track_id))
            outputs.append([x1, y1, x2, y2, track_id, track.stateOut, track.noConsider, track.trace])

            # calculate persons in and out
            if orientation in ["horizontal", "vertical"]:
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if self.is_vertical:
                    center_offset = center[0]
                else:
                    center_offset = center[1]

                if track.stateOut == 1 and (final_offset - center_offset < 0) and not track.noConsider:
                    personIn += 1
                    track.stateOut = 0
                    # track.noConsider = True  # I think this attribute should not be used, #yes, I agree.
                    # because it will miss those persons who turn around

                if track.stateOut == 0 and (final_offset - center_offset >= 0) and not track.noConsider:
                    personOut += 1
                    track.stateOut = 1
                    # track.noConsider = True
            elif orientation == "in_out":
                coord = (x1, y1, x2, y2)
                sign = self.judge_in_out(coord, boundary)
                if track.stateOut == 1 and not sign:
                    personOut += 1
                    track.stateOut = 0
                elif track.stateOut == 0 and sign:
                    personIn += 1
                    track.stateOut = 1

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs, personOut, personIn

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])

        return features
