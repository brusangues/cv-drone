# USAGE__________________________________________________________________________
# python YOLO_final.py -i data/100.JPEG -o output -y yolo-custom -b 0 -d 1 -r 416
# python YOLO_final.py -i data/hubsan.mp4 -o output -y yolo-custom -b 0 -d 1 -r 416
# python YOLO_final.py -i data/test.mp4 -o output -y yolo-custom -b 0 -d 1 -r 416
# python YOLO_final.py -i data/10.JPEG -o output -y yolo-custom -b 0 -d 1 -r 416

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime
import glob
import re
import math
from yolo_utils import *

# ENTRADAS__________________________________________________________________________
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="data/100.JPEG",
                help="path to input image or video")
ap.add_argument("-o", "--output", default="output",
                help="folder to output image or video")
ap.add_argument("-y", "--yolo", default="yolo-custom",
                help="base path to YOLO directory")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
ap.add_argument("-m", "--max", type=int, default=5,
                help="maximum number of detections per frame")
ap.add_argument("-b", "--blob", type=int, default=0,
                help="display or not blob")
ap.add_argument("-d", "--debug", type=int, default=0,
                help="print information to debug")
ap.add_argument("-r", "--resolution", type=int, default=416,
                help="rxr resolution for yolo. 320, 416, 608, 832")
args = vars(ap.parse_args())


# INICIALIZAÇÃO_______________________________________________________________________
# PATHS
inputIsImage = re.search( r'/.+\.(jpg|JPEG|tiff|png)$', args["input"], re.I)
inputIsVideo = re.search( r'/.+\.(mp4|avi|mkv)$', args["input"], re.I)

if inputIsImage:
    print("[INFO] input is image")
    outputPath = args["output"]+inputIsImage.group()
    #print(outputPath)
elif inputIsVideo:
    print("[INFO] input is video")
    outputPath = args["output"]+inputIsVideo.group()
    #print(outputPath)
else:
    print("[ERROR] input not supported")
    exit()

# LABELS
# load the COCO class labels our YOLO model was trained on

labelsPath = glob.glob(args["yolo"]+"/*.names")[0]
LABELS = open(labelsPath).read().strip().split("\n")

# COLORS
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# COLORS: list of colors

#WEIGHTS and CONFIG
weightsPath = glob.glob(args["yolo"]+"/*.weights")[0]
configPath = glob.glob(args["yolo"]+"/*.cfg")[0]
if bool(args["debug"]): print(configPath)
if bool(args["debug"]): print(weightsPath)

# YOLO
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
yoloNet = Yolov3(weightsPath, configPath, args["resolution"], args["confidence"], args["threshold"])

# VIDEO STREAM
# initialize the video stream, pointer to output video file, and dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

#FRAMES and FPS
# try to determine the total number of frames in the video file and fps
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
    #vs.set(cv2.CAP_PROP_FPS, 2000)
    fps = vs.get(cv2.CAP_PROP_FPS)
    if fps == 1000:
        fps = 1
    print("[INFO] {} FPS in video".format(fps))
except:
    # an error occurred while trying to determine the total
    # number of frames in the video file
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# LOOP_______________________________________________________________________________________
# loop over frames from the video file stream
elap_avg = []
while True:
    start = time.time()
    # NEXT FRAME If grabbed is False, end of stream.
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    # DIMENSIONS If the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        print("[INFO] height and width ", H, W)

    # PROCESSAMENTO_______________________________________________________________________________

    # BLOB
    blob = yoloNet.blobFromImage(frame)

    # FORWARD PASS YOLO
    idxs, boxes, confidences, classIDs = yoloNet.forwardPass(blob)

    # BOUNDING BOXES
    
    boundingBoxThickness = math.ceil((H+W)/800)
    
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten()[:args["max"]]:
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, boundingBoxThickness)
            
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, boundingBoxThickness/2, color, 1)

    # DISPLAY________________________________________________________________________________
    # Resize the frame and convert it to grayscale (while still
    # retaining 3 channels) - possivelmente maior rapidez de exibição
    #frame = imutils.resize(frame, width=800)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = np.dstack([frame, frame, frame])

    # TIME CALCULATIONS
    end = time.time()
    elap = (end - start)
    elap_avg.append(elap)
    
    textX = boundingBoxThickness*5; textY = textX*2+3
    text1 = "Frame time: {:.4f} s".format(elap)
    text2 = "Frame avrg: {:.4f} s".format(np.mean(elap_avg))
    cv2.putText(frame, text1, (textX, textY),
                cv2.FONT_HERSHEY_SIMPLEX, boundingBoxThickness/2, [255, 0, 0], boundingBoxThickness)
    cv2.putText(frame, text2, (textX, textY*2),
                cv2.FONT_HERSHEY_SIMPLEX, boundingBoxThickness/2, [255, 0, 0], boundingBoxThickness)

    # show the frame and update the FPS counter
    if args["blob"]:
        blob_img = np.zeros((args["resolution"], args["resolution"], 3))
        blob_img[:, :, 0] = blob[0, 0, :, :]
        blob_img[:, :, 1] = blob[0, 1, :, :]
        blob_img[:, :, 2] = blob[0, 2, :, :]
        cv2.putText(blob_img, "Blob", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 2, 3)
        frame = img_glue(frame, blob_img)

    # OUTPUT______________________________________________________________________
    show = imutils.resize(frame.copy(), width=800)
    show = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
    #show = np.dstack([show, show, show)
    cv2.imshow("Frame", show)
    
    cv2.waitKey(1)
    if chr(cv2.waitKey(50)&255) == 'q':
        break

    if inputIsImage:
        print("[INFO] image saved to "+outputPath)
        cv2.imwrite(outputPath, frame)
        time.sleep(3)
        break
    
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG") #MPEG FMP4
        writer = cv2.VideoWriter(outputPath, fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
    # write the output frame to disk
    if inputIsVideo:
        writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
if inputIsVideo:
    writer.release()
vs.release()
print("[INFO] exit")