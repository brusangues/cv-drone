import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime
import glob

# yoloNet = Yolov3(weightsPath, configPath, args["resolution"], args["confidence"], args["threshold"])

class Yolov3:
    def __init__(self, weightsPath, configPath, resolution, confidence, threshold):
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #net: network object
        self.ln = self.net.getLayerNames() #ln: List of names of neurons like 'conv_0', 'bn_0' 
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
	        #ln:list of output layers like ['yolo_82', 'yolo_94', 'yolo_106']
        
        self.resolution = resolution
        self.confidence = confidence
        self.threshold  = threshold

# blob = yoloNet.blobFromImage(frame)
    def blobFromImage(self, frame):
        (self.H, self.W) = frame.shape[:2]
        
        #BLOB
        #Construct a blob from the input frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.resolution, self.resolution), swapRB=True, crop=False) 
            #blob.type = np.darray (nimages,ncolors,H,W)
        return blob

# idxs, boxes, confidences, classIDs = yoloNet.forwardPass(blob)
    def forwardPass(self, blob):
        #FORWARD PASS YOLO
        self.net.setInput(blob[:3]) #Sets the new value for the layer output blob.
        layerOutputs = self.net.forward(self.ln) #Runs forward pass for the whole network
            #layerOutputs: list of lists of detections

        #OUTPUT TREATMENT
        boxes = []
        confidences = []
        classIDs = []
        #Loop over each of the layer outputs
        for output in layerOutputs:
            #Loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
        #NON MAXIMA SUPPRESSION
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        return idxs, boxes, confidences, classIDs

def img_glue(img1, img2):
	img2 = img2.astype(np.float64) / np.max(img2) # normalize the data to 0 - 1
	img2 = 255 * img2 # Now scale by 255
	img2 = img2.astype(np.uint8)
	y1,x1 = img1.shape[:2]
	y2,x2 = img2.shape[:2]
	img3 = np.zeros([max(y1,y2), x1+x2, 3], dtype=np.uint8)
	img3[:y1, :x1, :] = img1
	img3[:y2, x1:x1+x2, :] = img2
	return img3