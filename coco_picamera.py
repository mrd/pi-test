#!/usr/bin/python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera
from picamera import Color
import pdb

from PIL import Image
from tflite_runtime.interpreter import Interpreter

def clamp(minvalue, value, maxvalue):
    return max(minvalue, min(value, maxvalue))

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_arrays = interpreter.get_output_details()
  output_details = output_arrays[0]
  #output = np.squeeze(interpreter.get_tensor(output_details['index']))
  output = []
  for i in range(4):
    output.append(np.squeeze(interpreter.get_tensor(output_arrays[i]['index'])))

  # If the model is quantized (uint8 data), then dequantize the results
  #if output_details['dtype'] == np.uint8:
    #scale, zero_point = output_details['quantization']
    #output = scale * (output - zero_point)

  #ordered = np.argpartition(-output, top_k)
  #return [(i, output[i]) for i in ordered[:top_k]]
  return output


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  cameraW = 640
  cameraH = 480
  def clampW(x):
    return clamp(0,x,cameraW-1)
  def clampH(x):
    return clamp(0,x,cameraH-1)
  ov = np.zeros((cameraH, cameraW, 3), dtype=np.uint8)
  ov[:, :, :] = 0

  frameTime = time.time()*1000
  overlayInterval = 100

  with picamera.PiCamera(resolution=(cameraW, cameraH), framerate=30) as camera:
    camera.start_preview(alpha=255)
    camera.annotate_foreground = Color('black')
    camera.annotate_background = Color('white')
    overlay = camera.add_overlay(ov.tobytes(), layer=3, alpha=64)
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        stream.seek(0)
        stream.truncate()
        bestIdx = np.argmax(results[2])
        label = labels[results[1][bestIdx]+1]
        prob = results[2][bestIdx]
        top = int(np.round(results[0][bestIdx][0]*cameraH))
        left = int(np.round(results[0][bestIdx][1]*cameraW))
        bottom = int(np.round(results[0][bestIdx][2]*cameraH))
        right = int(np.round(results[0][bestIdx][3]*cameraW))
        ov[:,:,:] = 0
        if top >= 0 and top < cameraH:
          ov[top, clampW(left):clampW(right), :] = 0xff
        if bottom >= 0 and bottom < cameraH:
          ov[bottom, clampW(left):clampW(right), :] = 0xff
        if left >= 0 and left < cameraW:
          ov[clampH(top):clampH(bottom), left, :] = 0xff
        if right >= 0 and right < cameraW:
          ov[clampH(top):clampH(bottom), right, :] = 0xff
        if time.time()*1000 - frameTime > overlayInterval:
          overlay.update(ov.tobytes())
          frameTime = time.time()*1000
        #pdb.set_trace()
        camera.annotate_text = '%s (%.1f%%)\n%.1fms' % (label, prob*100, elapsed_ms)
    finally:
      camera.remove_overlay(overlay)
      camera.stop_preview()


if __name__ == '__main__':
  main()
