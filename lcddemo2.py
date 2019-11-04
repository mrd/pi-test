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

#import LCD_1in8
#import LCD_Config
from st7735_ijl20.st7735 import ST7735

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor

import argparse
import io
import time
import numpy as np
import signal
import picamera
from picamera import Color

from tflite_runtime.interpreter import Interpreter


class SigTerm(SystemExit): pass
def sigterm(sig,frm): raise SigTerm
signal.signal(15,sigterm)

FONT_SMALL = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 10)
FONT_LARGE = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 12)

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


DISPLAY_WIDTH = 160                # LCD panel width in pixels
DISPLAY_HEIGHT = 128               # LCD panel height
def main():
  ##################################################
  # Initialise LCD
  ## LCD = LCD_1in8.LCD()
  LCD = ST7735()
  ## Lcd_ScanDir = LCD_1in8.SCAN_DIR_DFT
  ## LCD.LCD_Init(Lcd_ScanDir)
  LCD.begin()
  ## screenbuf = Image.new("RGB", (LCD.LCD_Dis_Column, LCD.LCD_Dis_Page), "WHITE")
  screenbuf = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), "WHITE")
  draw = ImageDraw.Draw(screenbuf)
  draw.text((33, 22), 'Initialising...', fill = "BLUE", font = FONT_LARGE)
  ## LCD.LCD_PageImage(screenbuf)
  LCD.display(screenbuf)

  ##################################################

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

  frameTime = time.time()*1000


  with picamera.PiCamera(resolution=(cameraW, cameraH), framerate=30) as camera:
    camera.start_preview(alpha=255)
    camera.annotate_foreground = Color('black')
    camera.annotate_background = Color('white')
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

        # paste camera captured image into screen buffer
        screenbuf.paste(image.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT)))

        availColours = ['salmon', 'olive', 'orange', 'purple', 'aqua', 'darkgray', 'yellow', 'sienna', 'red', 'blue', 'green']
        usedColours = {}
        msg = ""
        for i in range(10):
            label = labels[results[1][i]+1]
            prob = clamp(0,results[2][i],1)
            if prob >= 0.5:
                if label in usedColours:
                    colour = usedColours[label]
                elif availColours == []:
                    colour = 'salmon'
                else:
                    colour = availColours.pop()
                    usedColours[label] = colour

                top = clamp(0,results[0][i][0],1)
                left = clamp(0,results[0][i][1],1)
                bottom = clamp(0,results[0][i][2],1)
                right = clamp(0,results[0][i][3],1)
                # draw bounding box
                draw.rectangle([(left*DISPLAY_WIDTH,top*DISPLAY_HEIGHT),(right*DISPLAY_WIDTH,bottom*DISPLAY_HEIGHT)]
                               , outline = colour)
                desc = ("{0} {1:3.1f}%".format(label,prob*100))
                txtw, txth = FONT_SMALL.getsize(desc)
                # draw label rectangle
                draw.rectangle([(left*DISPLAY_WIDTH,top*DISPLAY_HEIGHT),(left*DISPLAY_WIDTH+txtw,top*DISPLAY_HEIGHT+txth)]
                               , fill = colour)
                # draw label
                draw.text((left*DISPLAY_WIDTH, top*DISPLAY_HEIGHT), desc, fill = "WHITE", font = FONT_SMALL)

                # record info for log
                msg += ("{0:20} {1:3.1f}% {2:3.3f} {3:3.3f} {4:3.3f} {5:3.3f} {6: 5.1f}ms\n".format(label,prob*100,top,left,bottom,right,elapsed_ms))


        # draw.text((0, 0), msg, fill = "BLUE", font = FONT_SMALL)

        LCD.display(screenbuf)
        msg += ("--------------------------------------------------\n")
        print(msg)

        #pdb.set_trace()

        bestIdx = np.argmax(results[2])
        label = labels[results[1][bestIdx]+1]
        prob = clamp(0,results[2][bestIdx],1)
        top = clamp(0,results[0][bestIdx][0],1)
        left = clamp(0,results[0][bestIdx][1],1)
        bottom = clamp(0,results[0][bestIdx][2],1)
        right = clamp(0,results[0][bestIdx][3],1)
        # camera.annotate_text = '%s (%.1f%%)\n%.1fms' % (label, prob*100, elapsed_ms)
    finally:
      camera.stop_preview()
      LCD.LCD_Clear()


if __name__ == '__main__':
  main()
