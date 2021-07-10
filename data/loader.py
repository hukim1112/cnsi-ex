import os
import numpy as np
import cv2
import tensorflow as tf
from pycocotools.coco import COCO
from functools import partial
from utils.iou import compute_iou

def object_bboxes(anns):
    object_boxes = []
    for ann in anns:
        x,y,w,h = ann["bbox"]
        x,y,w,h = x/2024,y/10000,w/2024,h/10000
        object_boxes.append([x,y,x+w,y+h])
    object_boxes = np.array(object_boxes, np.float32)
    return object_boxes

def object_segmentations(anns):
    joints = []
    joint_gaps = []
    for ann in anns:
        for seg in ann["segmentation"]:
            if ann["category_id"] == 0: #joint
                joints.append(seg)
            else: #gap
                joint_gaps.append(seg)
    return joints, joint_gaps

def object_masking(mask_array, segmentations, color):
    if len(segmentations) != 0:
        for seg in segmentations:
            pts = []
            for i in range(int(len(seg)/2)):
                pts.append([int(seg[i*2]), int(seg[i*2+1])])
            pts_np = np.array(pts, np.int32)
            if pts_np.size != 0:
                cv2.fillPoly(mask_array, [pts_np], color)

def detection_pipeline(image_array, object_boxes):
    image_array = image_array[tf.newaxis, ...] #add a axis(the number of data) for tensorflow function
    patches = tf.image.extract_patches(images=image_array,
                           sizes=[1, 1000, 1024, 1],
                           strides=[1, 500, 1, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
    #extract patches from image array. the size of each patch is (1000, 1024) and stride step is 500 pixels.
    patches = tf.squeeze(patches)
    patches = tf.reshape(patches, (19, 1000,1024,3))
    patches = tf.cast(patches, tf.int32)
    boxes = tf.constant([[0.  , 0.  , 1.  , 0.1 ],
                        [0.  , 0.05, 1.  , 0.15],
                        [0.  , 0.1 , 1.  , 0.2 ],
                        [0.  , 0.15, 1.  , 0.25],
                        [0.  , 0.2 , 1.  , 0.3 ],
                        [0.  , 0.25, 1.  , 0.35],
                        [0.  , 0.3 , 1.  , 0.4 ],
                        [0.  , 0.35, 1.  , 0.45],
                        [0.  , 0.4 , 1.  , 0.5 ],
                        [0.  , 0.45, 1.  , 0.55],
                        [0.  , 0.5 , 1.  , 0.6 ],
                        [0.  , 0.55, 1.  , 0.65],
                        [0.  , 0.6 , 1.  , 0.7 ],
                        [0.  , 0.65, 1.  , 0.75],
                        [0.  , 0.7 , 1.  , 0.8 ],
                        [0.  , 0.75, 1.  , 0.85],
                        [0.  , 0.8 , 1.  , 0.9 ],
                        [0.  , 0.85, 1.  , 0.95],
                        [0.  , 0.9 , 1.  , 1.  ]], tf.float32)
    ious = compute_iou(object_boxes, boxes)
    ious = tf.reduce_sum(ious, axis=0)
    ious = tf.squeeze(ious)
    labels = tf.where(ious>0, 1, 0)
    return patches, labels

def segmentation_pipeline(mask_array, labels, segmentations):
    object_masking(mask_array, segmentations[0], 1) #joint
    object_masking(mask_array, segmentations[1], 2) #joint_gap
    indice = tf.squeeze(tf.where(labels))
    select = tf.random.categorical(tf.math.log([tf.ones_like(indice)/len(indice)]), 1)
    idx = indice[select[0,0]]
    idx = tf.cast(idx, tf.int32)
    segmentation_mask = mask_array[500*idx:1000+500*idx,:]
    return segmentation_mask, idx

def data_process(image_id, coco, image_path):
    image_id = int(image_id.numpy())
    image = coco.loadImgs(image_id)
    file = image_path + image[0]["path"]
    image_array = cv2.imread(file)[:,:,::-1]
    mask_array = np.zeros_like(image_array, np.int32)
    anno_ids = coco.getAnnIds(imgIds=image_id, catIds=[0,1], iscrowd=None)
    anns = coco.loadAnns(anno_ids)

    object_boxes = object_bboxes(anns)
    patches, labels = detection_pipeline(image_array, object_boxes)
    joints, joint_gaps = object_segmentations(anns)
    segmentation_mask, idx = segmentation_pipeline(mask_array, labels, [joints, joint_gaps])
    return patches, labels, segmentation_mask, patches[idx]

def load(image_ids, coco, img_path):
    ds = tf.data.Dataset.from_tensor_slices(image_ids)
    _data_process = partial(data_process, coco = coco, image_path=img_path)
    ds = ds.map(lambda x: tf.py_function(_data_process, [x], [tf.int32, tf.int32, tf.int32, tf.int32]))
    return ds
