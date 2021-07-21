import tensorflow as tf
import cv2

class Inference():
    def __init__(self, detection_model, segmentation_model, det_thresh=0.5, seg_thresh=0.5):
        self.detection_model = detection_model
        self.segmentation_model = segmentation_model
        self.det_thresh = det_thresh
        self.seg_thresh = seg_thresh
    def detect(self, patches):
        patches = tf.keras.applications.mobilenet_v2.preprocess_input(patches)
        patches = tf.image.resize(patches, [224, 224])
        det_result = self.detection_model(patches, training=False)
        idx = tf.math.argmax(tf.where(det_result>self.det_thresh, det_result, 0)[:,1])
        return idx
    def segment(self, patch):
        patch = patch[tf.newaxis,...]
        resized_patch = tf.image.resize(patch, [512,512], method='nearest')
        resized_patch = resized_patch/255.
        seg_result = self.segmentation_model(resized_patch, training=False)
        seg_result = tf.where(seg_result>self.seg_thresh, 1, 0)
        return seg_result
    def __call__(self, image_array):
        image_array = image_array[tf.newaxis,...]
        H,W = image_array.shape[1:3]
        patches = tf.image.extract_patches(images=image_array,
                           sizes=[1, int(H/10), W, 1],
                           strides=[1, int(H/20), 1, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
        patches = tf.squeeze(patches)
        patches = tf.reshape(patches, (19, int(H/10),W,3))
        patches = tf.cast(patches, tf.float32)
        idx = self.detect(patches)
        patch = patches[idx]
        h,w = patch.shape[:2]
        seg_result = self.segment(patch)
        seg_result = tf.image.resize(seg_result, [h,w], method="nearest")
        return idx, patch, seg_result[0]
