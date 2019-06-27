#!/usr/bin/env python3

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import numpy as np
import os
import errno
import sys
from object_detector import ObjectDetector as TFObjectDetector
import drawing_utils
import cv2

description_text = """\
Use this script to visualize network output on each frame of a video.

Once you've trained a network, you may want to intuitively understand its
performance on different videos, especially looking at frame to frame
performance in a single video. This script enables that visualization for both
TensorFlow and TFLite model formats. Additionally, this script lets you save a
video with each frame annotated with output from the network, as well as save
individual annotated frames if desired.
"""

epilog_text = """\
example:
    ./camera_cv.py --movie [movie.mp4] --path_to_model [model.pb]
"""

parser = argparse.ArgumentParser(
        description=description_text,
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--movie", type=str, default="",
        help="Movie file to run prediction on")
parser.add_argument("--write_images", default=False, action="store_true",
        help="Whether to write each frame as a separate image")
parser.add_argument("--write_movie", default=False, action="store_true",
        help="Whether to write an annotated movie")
parser.add_argument("--path_to_model", type=str,
        default="output_inference_graph/frozen_inference_graph.pb",
        help="Directory containing frozen checkpoint file or .tflite model")
parser.add_argument("--path_to_labels", type=str,
        default="train_data/label.pbtxt",
        help="Text proto (TF) or text (tflite) file containing label map")
parser.add_argument("--threshold", type=float, default=0.6,
        help="Threshold for displaying detections")
parser.add_argument("--headless", default=False, action="store_true",
        help="Don't display anything when running script")
parser.add_argument("--trt", default=False, action="store_true",
        help="Whether the model was optimized with TRT")
args = parser.parse_args()

if args.movie is not "" and not os.path.exists(args.movie):
    print("Movie file %s missing" % args.movie)
    sys.exit(1)

if args.movie is not "":
    cam = cv2.VideoCapture(args.movie)
else:
    cam = cv2.VideoCapture(0)
    args.movie = "movie.mkv"


width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
objdet = TFObjectDetector(args.path_to_model, args.path_to_labels, args.trt)

model_name = os.path.splitext(os.path.basename(args.path_to_model))[0]
movie_name = os.path.splitext(os.path.basename(args.movie))[0]

if args.write_movie:
    out_path = os.path.join(os.path.dirname(args.movie),
            "_".join([movie_name, model_name, "boxes"]))
    movie_path = "%s.mkv" % out_path
    print("Writing movie to", movie_path)
    writer = cv2.VideoWriter(
            movie_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            int(cam.get(cv2.CAP_PROP_FPS)),
            (width, height)
    )

    # Quit if there was a problem
    if not writer.isOpened():
        print("Unable to open video!")
        sys.exit()

if args.write_images:
    movie_dir = os.path.dirname(args.movie)
    images_dir = os.path.join(movie_dir, "%s_images" % movie_name)
    print("Writing images to %s" % images_dir)

    try:
        os.makedirs(images_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("Directory exists already, continuing!")
        else:
            raise

counter = 0

ret, frame = cam.read()
while ret == True:
    img = frame.copy() # Aliased, but lets us turn off transformations as necessary.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    expand = np.expand_dims(img, axis=0)
   
    # Time how long detections take, roughly.
    det_start = time.time()
    result = objdet.detect(expand)
    det_end = time.time()

    boxes = []
    for i in range(result['num_detections']):
        if result['detection_scores'][i] > args.threshold:
            class_ = result['detection_classes'][i]
            box = result['detection_boxes'][i]
            score = result['detection_scores'][i]
            y1, x1 = int(box[0] * h), int(box[1] * w)
            y2, x2 = int(box[2] * h), int(box[3] * w)
            boxes.append((class_, score, x1, y1, x2, y2))

    bboxes = [np.array([x1, y1, x2-x1, y2-y1]) for
                (cls, score, x1, y1, x2, y2) in boxes]
    classes = [objdet.category_index[int(cls)]['name'] for cls, *_ in boxes]
    drawing_utils.draw_bboxes(img, bboxes, classes)

    print("Frame:", counter, end="\r")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    drawing_utils.shadow_text(img, "Frame Rate: %0.2f" %
            (1.0 / (det_end - det_start)), (0, 20))

    if not args.headless:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args.write_movie:
        writer.write(img)

    if args.write_images:
        print("[%d] Writing original to %s" % (counter, images_dir))
        cv2.imwrite(os.path.join(images_dir, "orig_%05d.png" % counter), frame)
        print("[%d] Writing boxes to %s" % (counter, images_dir))
        cv2.imwrite(os.path.join(images_dir, "box_%05d.png" % counter), img)
    counter += 1
    ret, frame = cam.read()

print("Done!")

if args.write_movie:
    writer.release()
