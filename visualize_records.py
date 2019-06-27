#!/usr/bin/env python3

import drawing_utils
import numpy as np
import os
import cv2
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("records", nargs="+",
        help="Path to records to decode")
parser.add_argument("-s", "--scale", type=float, default=1.0,
        help="Amount to scale output window by")

args = parser.parse_args()

def main():

    example = tf.train.Example()
    for record_path in args.records:
        if not os.path.isfile(record_path):
            print("Record %s does not exist!" % record_path)
            continue
    
        for record in tf.io.tf_record_iterator(record_path):
            example.ParseFromString(record)
            features = example.features.feature # dict mapping
            
            image_bytes = np.frombuffer(
                    features["image/encoded"].bytes_list.value[0],
                    dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)


            im_height, im_width, _ = image.shape
            xmin = np.array(
                    features['image/object/bbox/xmin'].float_list.value) * \
                            im_width
            xmax = np.array(
                    features['image/object/bbox/xmax'].float_list.value) * \
                            im_width
            ymin = np.array(
                    features['image/object/bbox/ymin'].float_list.value) * \
                            im_height
            ymax = np.array(
                    features['image/object/bbox/ymax'].float_list.value) * \
                            im_height
            
            text = features['image/object/class/text'].bytes_list.value
            text = [text[i].decode('utf-8') for i in range(len(text))]
            
            width = xmax - xmin
            height = ymax - ymin

            bboxes = np.column_stack((xmin, ymin, width, height))
            print("Bounding Boxes:", bboxes)

            image = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale)

            scaled_bboxes = bboxes * args.scale
            drawing_utils.draw_bboxes(image, scaled_bboxes, text)
         
            lines = []
            filename = features['image/filename'].bytes_list.value[0]
            filename = filename.decode('utf-8')
            filename = "/".join(filename.split("/")[-3:])
            lines.append("Record: %s" % record_path)
            lines.append(filename)
            lines.append("Num boxes: %d" % len(text))
            for i, line in enumerate(lines):
                drawing_utils.shadow_text(image, line, (5, 20 * i + 15),
                        font_scale=0.5, font_weight=1)

            cv2.imshow("window", image)
            if (cv2.waitKey(0) & 0xFF) == ord('q'):
                return
            

if __name__ == "__main__":
    main()
