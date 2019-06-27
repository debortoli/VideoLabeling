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

import cv2
import numpy as np

# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (255, 255, 255),
    (0, 0, 0),
]


def scale_bboxes(bboxes, sf):

    scaled_bboxes = []

    for bbox in bboxes:
        p0 = bbox[:2].astype(float)
        p1 = p0 + bbox[2:].astype(float)
        size = p1 - p0
        center = p0 + (size / 2)

        new_size = sf * size
        p0 = center - new_size / 2
        p1 = center + new_size / 2

        scaled_bboxes.append(np.array([p0, p1 - p0]).reshape(-1))

    return scaled_bboxes


# bboxes are [x, y, w, h]
def draw_bboxes(frame, bboxes, classes, scale=1):
    assert len(bboxes) == len(classes)

    scaled_bboxes = scale_bboxes(bboxes, 1 / scale)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        cls = classes[i]

        if bbox is None or cls is None:
            continue

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 255), 2, 1)
        cv2.putText(
            frame,
            cls,
            (int(bbox[0]), int(bbox[1] + bbox[3] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            2,
        )

        if scale != 1:
            bbox = scaled_bboxes[i]
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 100), 2, 1)


def draw_dots(frame, bboxes):
    # Draw dots on all four forners of bboxes
    node_color = (255, 255, 255)
    for bbox in bboxes:
        if bbox is not None:
            p0 = bbox[:2]
            p1 = p0 + bbox[2:]
            p2 = p0 + [bbox[2], 0]
            p3 = p0 + [0, bbox[3]]
            cv2.circle(
                frame, tuple(p0.astype(int)), 10, node_color, thickness=-1
            )
            cv2.circle(
                frame, tuple(p1.astype(int)), 10, node_color, thickness=-1
            )
            cv2.circle(
                frame, tuple(p2.astype(int)), 10, node_color, thickness=-1
            )
            cv2.circle(
                frame, tuple(p3.astype(int)), 10, node_color, thickness=-1
            )


def shadow_text(frame, text, loc, font_scale=0.5, font_weight=2):
    shadow_color = (0, 0, 0)
    shadow_loc = tuple(np.array(loc) + 2)
    font_color = (255, 255, 255)
    font_type = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        frame,
        text,
        shadow_loc,
        font_type,
        font_scale,
        shadow_color,
        font_weight,
    )
    cv2.putText(
        frame, text, loc, font_type, font_scale, font_color, font_weight
    )
