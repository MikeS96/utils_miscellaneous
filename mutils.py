# coding: utf-8
import ast
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
import cv2
import time
import numpy as np
import os
import sys
from Utils import server
from datetime import datetime
sys.path.append("..")


def engine_creation(eng):
    with open(eng, 'rb') as f:
        buf = f.read()
        engine = runtime.deserialize_cuda_engine(buf)
    return engine


def buffer_initialization(engine):
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(
            binding)) * engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    context = engine.create_execution_context()
    return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, stream, context


def detections_list(output, img, threshold, layout=7):
    height, width, channels = img.shape
    detections = []
    for i in range(int(len(output)/layout)):
        prefix = i*layout
        score = output[prefix+2]
        if score > threshold:
            index = int(output[prefix+0])
            classes = int(output[prefix+1])
            xmin = int(output[prefix+3]*width)
            ymin = int(output[prefix+4]*height)
            xmax = int(output[prefix+5]*width)
            ymax = int(output[prefix+6]*height)
            w = xmax-xmin
            h = ymax-ymin
            centroid = get_centroid(xmin, ymin, w, h)
            detections.append(((xmin, ymin, w, h), centroid, classes, score))
    return detections


def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # dims are equal to 300 x 300
    image = (2.0/255.0) * image - 1.0
    image = image.transpose((2, 0, 1))
    return image


def detection_pipeline(img, host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, stream, context):
    np.copyto(host_inputs[0], img.ravel())
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    output = host_outputs[0]
    return output

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, axis=0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
