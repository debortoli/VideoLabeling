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

import cv2
import numpy as np
import sys
import time
import pdb
import os
import bbox_writer
import multiprocessing
import argparse
import drawing_utils

description_text="""\
Note: This script is still in development and has some bugs. If you find any bugs, please
    contact Valmiki Kothare on the subt slack, or at valmiki.kothare.vk@gmail.com

Use this script to label a video with assisted 

Like labeler.py, this script lets you label a video for the various objects of interest, but
includes assisted tracking to speed up the process.
Recommended procedure:
    Use 'n' key to advance to first object in video
    Label the object as you would with labeler.py
    Press 'n' again to advance to next frame, which should have updated the bounding box to
        track the object
    When the object leaves the frame, the bounding box should be automatically removed.
        If it is not, simply manually remove it with 'x'(last bbox) or 'c'(all bboxes)
    If you make a mistake, go back, remove a mistaken label, and relabel manually
    Bounding boxes may not grow with an object as the camera comes closer, so use 'e' when the
        video is paused to enter correction mode, which will allow you to resize the bbox

Normal Mode Keybinds:

    SpaceBar: Toggle autoplay (step through frames without pausing to label)
    j:        Cut autoplay delay in half
    k:        Double autoplay delay
    n:        Go to the next frame which will be saved, and pause to label
    g:        Step backward one frame
    l:        Step forward one frame
    d:        Toggle rotation 180 degrees
    w:        Toggle validation mode (don't write out new images or labels)
    q:        Quit the labeler

   new:
    e:        Correction mode, shift bounding boxes if necessary(only use when paused)
    m:        Toggle backward tracking mode, use if you wish to find the object and track it
              backwards through the video. Only use if you have not labeled the previous frames
    v:        Use to go back to the previous save frame, useful for backward tracking



Label Mode Keybinds:

    Shift + [a-z]: Set [a-z] as the current class
    c:             Clear all bounding boxes
    x:             Clear the most recent bounding box
    y:             Load the last set of bounding boxes
    Mouse:         Draw bounding boxes

In normal usage, you'll likely want to follow a workflow like this:

    1. Enable autoplay (SpaceBar), disabling (also SpaceBar) when you find a
    video segment which contains any of the objects of interest.
    2. If you paused at the wrong frame, use 'g' an 'l' to step by a single
    frame until you find the right frame.
    3. Press 'n' to skip forward to the next saved frame.
    4. Label the current saved frame.
    5. Repeat steps 3 and 4 until the objects of interest are no longer visible.
    6. Go back to step 1, using 'j' and 'k' to adjust autoplay speed as desired.

Note that if you run this script multiple times on the same video, any previous
labels will be loaded. This lets you take a break from labeling, and also lets
you add new labels later if desired.
"""

parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("filename", type=argparse.FileType('r'))
parser.add_argument("-f", "--frames", type=int,
        help="Number of steps between each frame to save.", default=10)
parser.add_argument("-p", "--run_path", default="",
        help="Directory for storing output. Pass in path to tracker folder.")
parser.add_argument("-r", "--rotate", action="store_true", default=False,
        help="Rotate image 180 degrees before displaying (saved as rotated).")
parser.add_argument("-v", "--validation", action="store_true", default=False,
        help="Turn on validation mode. No new images or labels will be saved"
        " unless they already exist.")
parser.add_argument("-s", "--scale", type=float, default=1.0, required=False,
        help="Scale factor to help the tracker along")
parser.add_argument("-w", "--window_scale", default=1.0, type=float,
        help="How much to scale image by before displaying (save original)")
parser.add_argument("-t", "--tracker", type=int, default=2, required=False,
        help="Index of tracker to use, [0-7]")
parser.add_argument("-y", "--yes", action="store_true", default=False,
        help="Skip initial bounding box validation")
parser.add_argument("-x", "--experiment", action="store_true", default=False,
        help="Don't write out any files")
parser.add_argument("-e", "--refine", action="store_true", default=False,
        help="Auto-refine bounding boxes during tracking (experimental)")
parser.add_argument("--mac", action="store_true", default=False,
        help="Auto-refine bounding boxes during tracking (experimental)")
parser.add_argument("-d", "--decimate", type=float, default=1.0,
        help="Scale factor for tracked image. Smaller means faster tracking")


args = parser.parse_args()

WINDOW = "Tracked Labeling"
WINDOW_SCALE = args.window_scale
CACHE_SIZE = 150 # 5 seconds worth of frames

if args.mac:
    try:
        tracker_fns = [cv2.legacy.TrackerCSRT_create]
    except:
        print("\nCould not import tracker, you may need to remove the --mac flag\n")
else:
    try:
        tracker_fns = [cv2.TrackerCSRT_create]
    except:
        print("\nCould not import tracker, you may need the --mac flag\n")



current_class = None
last_bboxes = []
last_classes = []
brightness = 1.0


def open_vid(path):
    # Open the video
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print("Unable to open video")
        sys.exit()
    return vid


def show_scaled(window, frame, sf=WINDOW_SCALE):
    #  f = np.clip(frame * brightness, 0, 255)
    cv2.imshow(window, cv2.resize(frame.astype(np.uint8), (0, 0), fx=sf, fy=sf))


def draw_text(image, text, location):
    drawing_utils.shadow_text(image, text, location, font_scale=.5,
            font_weight=1)


def draw_frame_text(frame, frame_text):
    for i, line in enumerate(frame_text):
        draw_text(frame, line, (5, i * 15 + 15))


# Make sure all bboxes are given as top left (x, y), and (dx, dy). Sometimes
# they may be specified by a different corner, so we need to reverse that.
def standardize_bbox(bbox):
    p0 = bbox[:2]
    p1 = p0 + bbox[2:]

    min_x = min(p0[0], p1[0])
    max_x = max(p0[0], p1[0])
    min_y = min(p0[1], p1[1])
    max_y = max(p0[1], p1[1])

    ret = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
    print("Standardized %s to %s" % (bbox, ret))

    return ret


# Let the user do some labeling. If they press any key that doesn't map to a
# useful thing here, we return it.
def label_frame(original, bboxes, new_bboxes, classes, frame_text, trackers):
    global last_bboxes, last_classes, current_class

    points = []
    shift_pressed = False

    def draw(frame):
        drawing_utils.draw_bboxes(frame, bboxes, classes)
        draw_frame_text(frame, frame_text + ["Current class: %s" %
            current_class])
        show_scaled(WINDOW, frame)


    def mouse_callback(event, x, y, flags, params):
        frame = params.copy() # Copy of original that we can afford to draw on
        h, w, c = frame.shape

        # (x, y) in original image coordinates
        x = int(x / WINDOW_SCALE)
        y = int(y / WINDOW_SCALE)

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(np.array([x, y]))

        if len(points) == 1: # Still drawing a rectangle
            cv2.rectangle(frame, tuple(points[0]), (x, y), (255, 255, 0), 1, 1)

        # If the mouse is moved, draw crosshairs
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0))
        cv2.line(frame, (0, y), (w, y), (255, 0, 0))

        if len(points) == 2: # We've got a rectangle
            bbox = np.array([points[0], points[1] - points[0]]).reshape(-1)
            bbox = standardize_bbox(bbox)

            cls = str(current_class)
            bboxes.append(bbox)
            new_bboxes.append(bbox)
            classes.append(cls)
            points.clear()

        draw(frame)


    cv2.setMouseCallback(WINDOW, mouse_callback, param=original)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 225 or key == 226: # shift seems to be platform dependent
            shift_pressed = True

        elif (key == ord('b') or (shift_pressed and key == ord('b')) or
                key == ord('B')):
            shift_pressed = False
            current_class = "backpack"
            draw(original.copy())

        elif (key == ord('o') or (shift_pressed and key == ord('o')) or
                key == ord('O')):
            shift_pressed = False
            current_class = "drill"
            draw(original.copy())

        elif (key == ord('i') or (shift_pressed and key == ord('i')) or
                key == ord('I')):
            shift_pressed = False
            current_class = "fire extinguisher"
            draw(original.copy())

        elif (key == ord('t') or (shift_pressed and key == ord('t')) or
                key == ord('T')):
            shift_pressed = False
            current_class = "vent"
            draw(original.copy())
        
        elif (key == ord('s') or (shift_pressed and key == ord('s')) or
                key == ord('S')):
            shift_pressed = False
            current_class = "survivor"
            draw(original.copy())       
        
        elif (key == ord('p') or (shift_pressed and key == ord('p')) or
                key == ord('P')):
            shift_pressed = False
            current_class = "cell phone"
            draw(original.copy())
        
        elif (key == ord('r') or (shift_pressed and key == ord('r')) or
                key == ord('R')):
            shift_pressed = False
            current_class = "rope"
            draw(original.copy())

        elif (key == ord('h') or (shift_pressed and key == ord('h')) or
                key == ord('H')):
            shift_pressed = False
            current_class = "helmet"
            draw(original.copy())

        elif (key == ord('z') or (shift_pressed and key == ord('z')) or
                key == ord('Z')):
            shift_pressed = False
            current_class = "helmet-light"
            draw(original.copy())

        elif (key == ord('a') or (shift_pressed and key == ord('a')) or
                key == ord('A')):
            shift_pressed = False
            current_class = "rope-non-bunched"
            draw(original.copy())

        elif key == ord('c'): # Clear everything
            bboxes.clear()
            new_bboxes.clear()
            classes.clear()
            trackers.clear()
            draw(original.copy())

        elif key == ord('y'): # Get the data from the last label session
            bboxes.clear()
            new_bboxes.clear()
            classes.clear()

            bboxes.extend(last_bboxes)
            new_bboxes.extend(last_bboxes)
            classes.extend(last_classes)
            draw(original.copy())

        elif key != 255: # Default return value from waitKey, keep labeling
            break


    # Only save if we have non-empty labels
    if bboxes:
        last_bboxes = bboxes

    if classes:
        last_classes = classes

    cv2.setMouseCallback(WINDOW, lambda *args: None)
    return key


def load_bboxes(frame_number, run_path):
    # Figure out which file we're trying to load. First, get the path of the
    # image file that we'd be saving against.
    bbox_filename = os.path.join(run_path, "%05d.txt" % frame_number)
    if os.path.isfile(bbox_filename):
        bboxes, classes = bbox_writer.read_bboxes(bbox_filename)
    else:
        # Not saved yet, so just return an empty list
        bboxes = []
        classes = []

    return bboxes, classes


def save_frame(frame, bboxes, classes, run_path, frame_number, validation):

    frame_path = os.path.join(run_path, "%05d.png" % frame_number)
    bbox_path = os.path.join(run_path, "%05d.txt" % frame_number)

    if validation:
        if os.path.isfile(frame_path):
            bbox_writer.write_bboxes(bboxes, classes, bbox_path)
        else:
            print("Image %d not found, skipping." % frame_number)
    else:
        if not os.path.isfile(frame_path):
            print("Saving frame %d to %s" % (frame_number, frame_path))
            cv2.imwrite(frame_path, frame)

        bbox_writer.write_bboxes(bboxes, classes, bbox_path)


def add_trackers(tracker_index, frame, bboxes, trackers):
    frame = scale_frame_for_tracking(frame)
    bboxes = scale_bboxes_for_tracking(bboxes)

    tracker_fn = tracker_fns[tracker_index]

    for i, bbox in enumerate(bboxes):
        tracker = tracker_fn()
        if args.mac:
            bbox = tuple(int(tup) for tup in bbox)
        ret = tracker.init(frame, tuple(bbox))
        if not ret:
            print("Unable to initialize tracker", i)
            continue
        else:
#            print("Successfully initialized tracker", i)
            trackers.append(tracker)

def scale_bboxes_for_tracking(bboxes):
    sf = args.decimate
    scaled_bboxes = [tuple(np.array(bbox) * sf) for bbox in bboxes]
    #print("Scaling for tracking:", bboxes, scaled_bboxes)
    return scaled_bboxes


def scale_frame_for_tracking(frame):
    sf = args.decimate
    scaled_frame = cv2.resize(frame, None, fx=sf, fy=sf)
    return scaled_frame

def unscale_bbox_for_tracking(bbox):
    sf = args.decimate
    out = tuple(np.array(bbox) / sf)
    return out

def refine_bboxes(bboxes, classes, frame, trackers):
    # Refine boxes and reinitialize trackers.
    # Boxes are refined to be as tight as possible to the object being tracked.
    # The tracker is then given the bbox which has been inflated by the original
    # scale factor, to preserve tracking quality.

    # Just in case the tracker is missing something, we scale even further to
    # determine our ROI.
    scaled_bboxes = drawing_utils.scale_bboxes(bboxes, 1.2)

    h, w, _ = frame.shape

    # Very much hard coded for our particular use case.
    for i, bbox in enumerate(scaled_bboxes):
        if bbox is None: continue

        # Grab the part that we care about.
        rounded_bbox = bbox.astype(int)
        top_left = rounded_bbox[:2]
        bottom_right = top_left + rounded_bbox[2:]
        xs = np.clip([top_left[0], bottom_right[0]], 0, w)
        ys = np.clip([top_left[1], bottom_right[1]], 0, h)

        roi = frame[ys[0]:ys[1], xs[0]:xs[1]]

        # Resize the roi to be a reasonable dimension to see
        # Make the smaller of the two dimensions a fixed size
        IMAGE_SIZE = 100
        roi_h, roi_w, _ = roi.shape
        sf = IMAGE_SIZE / min(roi_h, roi_w)
        roi = cv2.resize(roi, (0, 0), fx=sf, fy=sf)

        new_bbox = None
        cls = classes[i]
        if cls == 'w':
            # TODO: Tune parameters here, if necessary
            print("Refining white whiffle ball")
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            min_radius = IMAGE_SIZE // 4
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                    minDist=IMAGE_SIZE/2, param1=30, param2=50,
                    minRadius=min_radius,
                    maxRadius=IMAGE_SIZE//2)
            if circles is None:
                print("NO CIRCLES DETECTED. UHHHH")
                continue

            # Find the biggest circle by area, aka biggest radius
            biggest_circle_index = np.argmax(circles[0, :, 2])
            biggest_circle = circles[0, biggest_circle_index]
            c = biggest_circle

            if (c[2] < min_radius):
                print("Got an invalid circle?")
                continue

            # draw the outer circle and a dot at the center
            cv2.circle(roi, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(roi, (c[0], c[1]), 2, (0, 0, 255), 3)

            # Use the bounding box of the circle to reinitialize the tracker.
            new_bbox = np.array([c[0] - c[2], c[1] - c[2], 2 * c[2], 2 * c[2]])


        elif cls == 'c':
            print("Refining orange cube")
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
            ret, thresh_h = cv2.threshold(hsv_blurred[:, :, 0], 30, 255,
                    cv2.THRESH_BINARY_INV)
            ret, thresh_s = cv2.threshold(hsv_blurred[:, :, 1], 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask = cv2.bitwise_and(thresh_h, thresh_s)


            # Clean up the mask a little
            kernel = np.ones((11,11),np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            #  cv2.imshow("Opening", opening)


            roi = cv2.bitwise_and(roi, roi, mask=mask)
            print("made the roi from the mask")

            # Grab the bounding box from the mask
            conn_stats = cv2.connectedComponentsWithStats(mask, connectivity=4)
            retval, labels, stats, centroids = conn_stats

            # The stats tell us [top left, top right, width, height, area]
            # Find the label with the biggest area
            if len(stats) > 1: # Means we have a non-bg label
                biggest_label = np.argmax(stats[1:, -1]) + 1

                p1 = stats[biggest_label, :2]
                p2 = p1 + stats[biggest_label, 2:-1]
                cv2.rectangle(roi, tuple(p1.astype(int)), tuple(p2.astype(int)), color=(255, 0, 100))
                print("drew the rectangle")

                new_bbox = stats[biggest_label, :-1]
        
        cv2.imshow("Image %d" % i, roi)

        if new_bbox is None:
            continue

        print("New bounding box", new_bbox)
        new_bbox = new_bbox / sf # Unscale by the same amount we scaled
        new_bbox = np.array([*(top_left + new_bbox[:2]), *new_bbox[2:]])

        print("Replacing bbox %d" % i, rounded_bbox, new_bbox)

        # Scale the bbox by the proper scale factor
        new_bbox_scaled = drawing_utils.scale_bboxes([new_bbox], args.scale)
        new_bbox_scaled = clamp_bboxes(new_bbox_scaled, w, h)

        # Force the scaled bounding box to be inside the bounds of the image.
        #  if any(new_bbox < 0):
        #      input()


        print("Initializing tracker")
        # Apply the new scaled bbox to both the tracker and the saved ones
        new_tracker = init_trackers(args.tracker, frame, new_bbox_scaled)[0]
        trackers[i] = new_tracker
        bboxes[i] = new_bbox_scaled[0]
        print("new scaled bbox", bboxes[i])


def init_trackers(tracker_index, frame, bboxes):
    frame = scale_frame_for_tracking(frame)
    bboxes = scale_bboxes_for_tracking(bboxes)

    trackers = []
    tracker_fn = tracker_fns[tracker_index]

    for i, bbox in enumerate(bboxes):
        tracker = tracker_fn()
        ret = tracker.init(frame, tuple(bbox))
        if not ret:
            print("Unable to initialize tracker", i)
            continue
        else:
            print("Successfully initialized tracker", i)
            trackers.append(tracker)

    return trackers


def correction_mode(orig, trackers, bboxes, classes):

    frame = orig.copy()
    drawing_utils.draw_bboxes(frame, bboxes, classes, args.scale)
    drawing_utils.draw_dots(frame, bboxes)

    show_scaled(WINDOW, frame)

    modified = set()
    tracked_box = None
    tracked_point = None

    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    POSITIONS = [TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]

    def mouse_callback(event, x, y, flags, params):
        nonlocal tracked_box, tracked_point

        orig, trackers, bboxes, classes = params
        im = orig.copy()

        # Determine which bbox is corresponded to by the click
        click = np.array([x, y]) / WINDOW_SCALE
        radius = 10

        # If there is no tracked point, determine which point gets clicked, if
        # any, and set variables accordingly.
        if tracked_point is None and event == cv2.EVENT_LBUTTONDOWN:
            for i, bbox in enumerate(bboxes):
                top_left = bbox[:2]
                top_right = top_left + [bbox[2], 0]
                bottom_left = top_left + [0, bbox[3]]
                bottom_right = top_left + bbox[2:]

                found = False
                for j, p in enumerate([top_left, top_right, bottom_left,
                        bottom_right]):
                    if np.linalg.norm(p - click) < radius:
                        tracked_point = POSITIONS[j]
                        tracked_box = i
                        modified.add(tracked_box)
                        found = True

                if found:
                    break

        elif tracked_point is not None and event == cv2.EVENT_LBUTTONDOWN:
            tracked_point = None
        elif tracked_point is not None:
            # There must be a tracked point, so move the point to the location
            # of the mouse click.
            p0 = bboxes[tracked_box][:2]
            p1 = p0 + bboxes[tracked_box][2:]

            if tracked_point == TOP_LEFT:
                p0 = click
            elif tracked_point == TOP_RIGHT:
                p0[1] = click[1]
                p1[0] = click[0]
            elif tracked_point == BOTTOM_LEFT:
                p0[0] = click[0]
                p1[1] = click[1]
            elif tracked_point == BOTTOM_RIGHT:
                p1 = click

            bboxes[tracked_box][:2] = p0
            bboxes[tracked_box][2:] = p1 - p0

        drawing_utils.draw_bboxes(im, bboxes, classes, args.scale)
        drawing_utils.draw_dots(im, bboxes)
        show_scaled(WINDOW, im)

    cv2.setMouseCallback(WINDOW, mouse_callback,
            param=(orig, trackers, bboxes, classes))

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord(' '):
            for mod in modified:
                print("Reinitializing tracker %d" % mod)
                new_tracker = init_trackers(args.tracker, orig, [bboxes[mod]])
                trackers[mod] = new_tracker[0]
            break
        elif key == ord('r'):
            refine_bboxes(bboxes, classes, orig, trackers)

    # Clear the mouse callbackcv2.setMouseCallback(window, lambda *args: None)
    cv2.setMouseCallback(WINDOW, lambda *args: None)


def check_bbox(bbox, frame):
    height = frame.shape[0]
    width = frame.shape[1]
    p0 = bbox[:2]
    p1 = []
    p1.append(p0[0] + bbox[2])
    p1.append(p0[1] + bbox[3])
#    print ("%d:%d , %d,%d" % (p0[0], p0[1], p1[0], p1[1]))
    if p0[0] < 5 or p0[1] < 5 or  p1[0] > width - 5 or p1[1] > height - 5:
        #print ("%d:%d , %d,%d" % (p0[0], p0[1], p1[0], p1[1]))
        return False
    else:
        return True


def main():
    vid = open_vid(args.filename.name)
    back_track = False
    prev_classes = []
    backward = False
    frame_classes = []
    trackers = []
    classes = []
    bboxes = []
    back_first = False
    tracker_index = args.tracker
    tracker_fn = tracker_fns[tracker_index]
    tracker_name = tracker_fn.__name__.split("_")[0]

    rotate_image = args.rotate
    validation = args.validation
    autoplay = False
    autoplay_delay = 32
    stop_at_next_save = False
    global brightness

    current_frame_number = 0
    last_removed_frame = -1

    stored_frames = dict()

    # Initialize the storage on disk
    filename = os.path.splitext(os.path.basename(args.filename.name))[0]

    if args.run_path == "":
        run_name = "%s" % (filename)
        run_path = os.path.join(os.path.dirname(args.filename.name), run_name)
        
        if not args.experiment: 
            try:
                os.mkdir(run_path)
            except:
                print("Directory probably exists already, continuing anyway.")
    else:
        run_path = args.run_path
        if not os.path.isdir(run_path):
            print("Run path %s is not a directory!" % run_path)
            return

    while True:
        is_save_frame = (args.frames > 0 and
                current_frame_number % args.frames == 0)
        new_bboxes = []
        temp_bboxes = []
        annotated_classes = []
        bboxes = []
#        print("%d:%d:%d" % (len(bboxes), len(classes), current_frame_number))

        if (current_frame_number not in stored_frames):
            ret, frame = vid.read()
            if not ret:
                print("Unable to open frame, quitting!")
                break


            # If this is a frame we care about, save it to disk. Also, see if
            # there is already a saved set of bboxes, and load those if they
            # exist.
            if validation:
                bboxes, classes = load_bboxes(current_frame_number, run_path)
            else:
                scaled_frame = scale_frame_for_tracking(frame)
                rem = []
                for i, tracker in enumerate(trackers):
                    ret, bbox = tracker.update(scaled_frame)
                    bbox = unscale_bbox_for_tracking(bbox)
                    if not ret:
                        bboxes.append(None)
                        rem.append(i)
                        print ("Tracking failure for %s" % classes[i])
                    elif not check_bbox(bbox, frame):
                        bboxes.append(None)
                        rem.append(i)
                        print ("%s lost" % classes[i])
                    else:
                        bboxes.append(np.array(bbox))
                        #annotated_classes.append("%d:%s" % (i, classes[i]))
                n = 0
                for i in rem:
                    j = i-n
                    bboxes.pop(j)
                    classes.pop(j)
                    trackers.pop(j)
                    n += 1

                if args.refine:
                    refine_bboxes(bboxes, classes, frame, trackers)

   
#            stored_frames[current_frame_number] = (frame.copy(), bboxes.copy(), classes.copy())

            if len(stored_frames) > CACHE_SIZE:
                last_removed_frame += 1
                stored_frames.pop(last_removed_frame)

        else:
            frame, temp_bboxes, temp_classes = stored_frames[current_frame_number]
            if (not validation) and ((back_track and not back_first) or not temp_bboxes):
                scaled_frame = scale_frame_for_tracking(frame)
                rem = []
                for i, tracker in enumerate(trackers):
                    ret, bbox = tracker.update(scaled_frame)
                    bbox = unscale_bbox_for_tracking(bbox)
                    if not ret:
                        bboxes.append(None)
                        rem.append(i)
                        print ("Tracking failure for %s" % classes[i])
                    elif not check_bbox(bbox, frame):
                        bboxes.append(None)
                        rem.append(i)
                        print ("%s lost" % classes[i])
                    else:
                        bboxes.append(np.array(bbox))
                        #annotated_classes.append("%d:%s" % (i, classes[i]))
                n = 0
                for i in rem:
                    j = i-n
                    bboxes.pop(j)
                    classes.pop(j)
                    trackers.pop(j)
                    n += 1
                if args.refine:
                    refine_bboxes(bboxes, classes, frame, trackers)
            elif back_first:
                back_first = False
                bboxes = temp_bboxes
                classes = temp_classes
            else:
                bboxes = temp_bboxes
                classes = temp_classes
                trackers.clear()
                add_trackers(tracker_index, frame.copy(), bboxes.copy(), trackers)
#            stored_frames[current_frame_number] = (frame.copy(), bboxes.copy(), classes.copy())


        if rotate_image:
            frame = frame[::-1, ::-1, :] # Makes a copy

        drawable_frame = frame.copy()
        frame_text = [
            "Frame number: " + str(current_frame_number) +
            (" (saved)" if is_save_frame else ""),
            "Autoplay: " + str(autoplay),
            "Autoplay delay: " + str(autoplay_delay),
            "Stopping at next save frame: " + str(stop_at_next_save),
            "Validation mode: " + str(validation),
            "Backward Tracking: " + str(back_track),
        ]
        draw_frame_text(drawable_frame, frame_text)
        # print("%d:%d:%d" % (len(bboxes), len(classes), current_frame_number))
        drawing_utils.draw_bboxes(drawable_frame, bboxes, classes)

        show_scaled(WINDOW, drawable_frame)

        if autoplay:
            delay = autoplay_delay if autoplay else 0
            key = cv2.waitKey(delay) & 0xFF
        elif backward:
            delay = autoplay_delay if backward else 0
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = label_frame(frame, bboxes, new_bboxes, classes, frame_text, trackers)
        if is_save_frame:
            save_frame(frame, bboxes, classes, run_path, current_frame_number,
                    validation)

            if stop_at_next_save:
                stop_at_next_save = False
                autoplay = False
                backward = False

        ##
        if not validation:
            add_trackers(tracker_index, frame.copy(), new_bboxes.copy(), trackers)
            
        stored_frames[current_frame_number] = (frame.copy(), bboxes.copy(), classes.copy())    
        original = frame.copy()
        # Handle whatever key the user pressed. The user may have potentially
        # labeled something, as above.
        if key == ord('q'):
            break
        elif key == ord('d'):
            rotate_image = not rotate_image
        elif key == ord('w'):
            validation = not validation
        elif key == ord('l'):
            current_frame_number += 1
            back_track = False
        elif key == ord('g'):
            current_frame_number -= 1
            current_frame_number = max(current_frame_number,
                    last_removed_frame + 1)
        elif key == ord('j'):
            autoplay_delay = max(autoplay_delay // 2, 1)
        elif key == ord('k'):
            autoplay_delay *= 2
        elif key == ord(' '):
            autoplay = not autoplay
            autoplay_delay = 32
            back_track = False
        elif key == ord('m'):
            back_track = not back_track
            back_first = True
        elif key == ord('v'):
            backward = True
            stop_at_next_save = True
            current_frame_number -= 1
        elif key == ord('e'):
            correction_mode(original, trackers, bboxes, classes)
            stored_frames[current_frame_number] = (original.copy(), bboxes.copy(), classes.copy())
        elif key == ord('n'):
            stop_at_next_save = True
            autoplay = True
            autoplay_delay = 1
            current_frame_number += 1
            back_track = False
        elif key == ord('+') or key == ord('='):
            brightness = min(brightness + 0.1, 3.0)
        elif key == ord('-') or key == ord('_'):
            brightness = max(brightness - 0.1, 0)
        elif autoplay:
            current_frame_number += 1
        elif backward:
            current_frame_number -= 1
        
        current_frame_number = max(current_frame_number,
                    last_removed_frame + 1) 
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
