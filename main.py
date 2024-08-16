import os 
import cv2 
from utils import get_detections
import numpy as np 
import random

config_path = '/Users/hanna m/machinelearning/deep_learning/cv/semantic_segmentaion/models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = '/Users/hanna m/machinelearning/deep_learning/cv/semantic_segmentaion/models/frozen_inference_graph.pb'
class_names = '/Users/hanna m/machinelearning/deep_learning/cv/semantic_segmentaion/models/coco.names'

img_path = '/Users/hanna m/machinelearning/deep_learning/cv/semantic_segmentaion/group.jpg'

image = cv2.imread(img_path)
h,w,c = image.shape

net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)

blob = cv2.dnn.blobFromImage(image)


boxes, masks = get_detections(net, blob)

empy_img= np.zeros((h, w, c))
colors = [[random.randint(0, 255),random.randint(0, 255), random.randint(0, 255)] for j in range(90)]

detection_threshold = 0.5 

for j in range(len(masks)):
    bbox = boxes[0, 0, j, :]

    class_id = bbox[1]
    score = bbox[2]

    if score > detection_threshold: 
      mask = masks[j]

      x1, y1, x2, y2 = int(bbox[3]*w), int(bbox[4]*h), int(bbox[5]*w), int(bbox[6]*h)
      mask = mask[int(class_id)]

      mask = cv2.resize(mask, ((x2-x1), (y2-y1)))

      _, thresh = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

      mask = thresh*255

      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0 ), 2)
      
      for c in range(3):
          
          empy_img[y1:y2, x1:x2, c] = mask*colors[int(class_id)][c]

overlay = ((0.6*empy_img) + 0.4*image).astype(np.uint8)

cv2.imshow('mask', image)
cv2.imshow('overlay', overlay)
cv2.waitKey(0)