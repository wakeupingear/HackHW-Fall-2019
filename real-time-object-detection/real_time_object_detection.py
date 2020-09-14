from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
#import os

import socket
import struct

os.environ["OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES"]="1"
os.environ["OPENCV_OPENCL_DEVICE"]=":GPU:1060"
    
TCP_IP = '127.0.0.1'
TCP_PORT = 8008

buffer_u16 = "H"
buffer_u8 = "b"

cv2.ocl.setUseOpenCL(True)
cv2.ocl.useOpenCL()
#cap0 = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture(1)

#print("Does my computer have opencl")
#print(cv2.ocl.haveOpenCL())
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prototxt", required=True,
#	help=r"C:\Users\CHANDLER_BLUE\Desktop\Hackathon_Fall_2019\real-time-object-detection")
#ap.add_argument("-m", "--model", required=True,
#	help=r"C:\Users\CHANDLER_BLUE\Desktop\Hackathon_Fall_2019\real-time-object-detection")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
#CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromCaffe(r"MobileNetSSD_deploy.prototxt.txt", r"MobileNetSSD_deploy.caffemodel")
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs2 = VideoStream(src=1).start()
time.sleep(2.0)
fps = FPS().start()
swap=0
yV=0

def fCheck(frame,x,xInt,yVal):
#    global yVal
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()

	# loop over the detections
    for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                 COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            if (swap==0)and(xInt==0):
                    yVal+=int((endX-startX)/2)+startX+(xInt*10000)
            if (idx==15):
                if (xInt==1 or xInt==0)and False:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((TCP_IP, TCP_PORT))
                    time.sleep(.00006)
                    data=int((endX-startX)/2)+startX+(xInt*10000)
#                    if swap==1: 
#                        data=round((data+yVal)/2)
#                        yVal=0
                    if xInt==0: print(data)
                    message = 50
                    buffer_type = buffer_u8+buffer_u16
                    buffer = struct.pack('<'+buffer_type,*[message,data])
                    s.send(buffer)
                    
                    
	# show the output frame
    cv2.imshow("Frame"+x, frame)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    fCheck(frame,"0",0,yV)
    
    #frame = vs2.read()
   # frame = imutils.resize(frame, width=400)
    #fCheck(frame,"1",1,yV)
	
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

	# update the FPS counter
    fps.update()
    if (swap==0): swap=1
    else: swap=0

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()