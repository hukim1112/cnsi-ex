import os
import numpy as np
import cv2
import tensorflow as tf
from pycocotools.coco import COCO
from functools import partial
from utils.iou import compute_iou

def object_bboxes(anns, H, W):
    object_boxes = []
    for ann in anns:
        x,y,w,h = ann["bbox"]
        x,y,w,h = x/W,y/H,w/W,h/H
        object_boxes.append([x,y,x+w,y+h])
    object_boxes = np.array(object_boxes, np.float32)
    return object_boxes

def calculate_patch_labels(object_boxes):
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
    return labels

def extract_patches(image_array, object_boxes, H, W):
    image_array = image_array[tf.newaxis, ...] #add a axis(the number of data) for tensorflow function
    patches = tf.image.extract_patches(images=image_array,
                           sizes=[1, int(H/10), W, 1],
                           strides=[1, int(H/20), 1, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
    #extract patches from image array. the size of each patch is (1000, 1024) and stride step is 500 pixels.
    patches = tf.squeeze(patches)
    patches = tf.reshape(patches, (19, int(H/10),W,3))
    patches = tf.cast(patches, tf.int32)
    labels = calculate_patch_labels(object_boxes)
    return patches, labels

def detection_process(image_id, coco, image_path):
    image_id = int(image_id.numpy())
    image = coco.loadImgs(image_id)
    file = image_path + image[0]["path"]
    image_array = cv2.imread(file)[:,:,::-1]
    H, W = image_array.shape[:2]

    anno_ids = coco.getAnnIds(imgIds=image_id, catIds=[0,1], iscrowd=None)
    anns = coco.loadAnns(anno_ids)

    object_boxes = object_bboxes(anns, H, W)
    patches, labels = extract_patches(image_array, object_boxes, H, W)
    patches = tf.image.resize(patches, [224, 224])
    patches = tf.cast(patches, tf.int32)
    return patches, labels

def load_detection(image_ids, coco, image_path, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(image_ids)
    if shuffle:
        ds = ds.shuffle(1000)
    map_func = partial(detection_process, coco = coco, image_path=image_path)
    ds = ds.map(lambda x: tf.py_function(map_func, [x], [tf.int32, tf.int32]))
    return ds

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

def extract_mask(image_array, labels, segmentations, H, W):
    mask_array = np.zeros([H,W])
    object_masking(mask_array, segmentations[0], 100) #joint
    object_masking(mask_array, segmentations[1], 200) #joint_gap
    indice = tf.squeeze(tf.where(labels), axis=1)
    select = tf.random.categorical(tf.math.log([tf.ones_like(indice)/len(indice)]), 1)
    idx = int(indice[select[0,0]])
    #idx = tf.cast(idx, tf.int32)
    patch = image_array[int(H/20)*idx:int(H/10)+int(H/20)*idx,:]
    mask = mask_array[int(H/20)*idx:int(H/10)+int(H/20)*idx,:]
    return patch, mask

def segmentation_process(image_id, coco, image_path):
    image_id = int(image_id.numpy())
    image = coco.loadImgs(image_id)
    file = image_path + image[0]["path"]
    image_array = cv2.imread(file)[:,:,::-1]
    H, W = image_array.shape[:2]

    anno_ids = coco.getAnnIds(imgIds=image_id, catIds=[0,1], iscrowd=None)
    anns = coco.loadAnns(anno_ids)
    joints, joint_gaps = object_segmentations(anns)

    object_boxes = object_bboxes(anns, H, W)
    labels = calculate_patch_labels(object_boxes)
    patch, mask = extract_mask(image_array, labels, [joints, joint_gaps], H, W)
    patch = cv2.resize(patch, [512,512], interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, [512,512], interpolation=cv2.INTER_NEAREST)
    #patches = tf.image.resize(patches, [224, 224])
    #patches = tf.cast(patches, tf.int32)
    return patch, mask

def load_segmentation(image_ids, coco, image_path, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(image_ids)
    if shuffle:
        ds = ds.shuffle(1000)
    map_func = partial(segmentation_process, coco = coco, image_path=image_path)
    ds = ds.map(lambda x: tf.py_function(map_func, [x], [tf.int32, tf.int32]))
    return ds
