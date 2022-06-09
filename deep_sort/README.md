


This repo refers to [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch), on top of that, this repo add some new features : 

1. count the number of persons in and out, and display it in Chinese characters
2. draw historical trajectory, and automatically delete those outdated trajectory. 
3. when there is a peson come in, the color of delimiter line will become from blue to red 



To finetune the tracking result, you can modify the `deep_sort/configs/deep_sort.yaml`

```
DEEPSORT:
  REID_CKPT: "deep_sort/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.2
  MIN_CONFIDENCE: 0.3
  NMS_MAX_OVERLAP: 0.5
  MAX_IOU_DISTANCE: 0.7
  MAX_AGE: 70
  N_INIT: 3
  NN_BUDGET: 100
```

REID_CKPT : Re-ID model path

MAX_DIST ：The matching threshold. Samples with larger distance are considered an invalid match. Used in `matching_cascade()`.

MIN_CONFIDENCE ： Those detection results whose confidence is lower than is value is ignored.

NMS_MAX_OVERLAP：ROIs that overlap more than this values are suppressed.

MAX_IOU_DISTANCE：Gating threshold. Associations with cost larger than this value are disregarded. Used in `min_cost_matching()`. 

MAX_AGE：Maximum number of missed frames before a track is deleted. 

N_INIT：Number of consecutive detections before the track is confirmed.

NN_BUDGET：Fix samples per class to at most to this number. Removes those oldest samples when the budget is reached.

**Reference** **Readings**

[《Deep SORT多目标跟踪算法代码解析(上)》](https://zhuanlan.zhihu.com/p/133678626)

[《Deep SORT多目标跟踪算法代码解析(下)》](https://zhuanlan.zhihu.com/p/133689982)