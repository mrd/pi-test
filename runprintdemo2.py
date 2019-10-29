#!/bin/bash
#python3 coco_detector2.py --model coco/2019-08-14-large.tflite --labels coco/labelmap.txt
python3 printdemo.py --model ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/detect.tflite --labels coco/labelmap.txt
