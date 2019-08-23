# USAGE
# python3 real_time_object_detection.py -p <prototxt> -m <caffemodel> -o <record_stream.mp4> -c <codec> -pc <picamera> -f <fps_value> -c <confidence_value>
# python3 real_time_object_detection.py

# Import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Construct the argument parser
ap = argparse.ArgumentParser()

# Arguments for real-time object detection
ap.add_argument("-p", "--prototxt", required=False, default='MobileNetSSD_deploy.prototxt.txt',
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default='MobileNetSSD_deploy.caffemodel',
	help="path to Caffe pre-trained model")
ap.add_argument("-conf", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
	
# Arguments for storing video
ap.add_argument("-o", "--output", required=False, default='recording.avi',
	help="path to output video file")
ap.add_argument("-pc", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
	
# Parse the arguments
args = vars(ap.parse_args())

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
	
# Generate a set of bounding box colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video stream, allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# Initialize the FPS counter
fps = FPS().start()

# Initialize the FourCC, VideoWriter, dimensions of the frame, and zeros array
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
(h, w) = (None, None)
zeros = None

# Loop over the frames from the video stream
while True:
	# Grab the frame from the threaded video stream
	frame = vs.read()
	# Resize frameto have a maximum width of 400 pixels
	frame = imutils.resize(frame, width=400)
	
	# Check if the writer is None
	if writer is None:
		# Grab the frame dimensions
		(h, w) = frame.shape[:2]
		# Store the image dimensions, initialize the video writer
		writer = cv2.VideoWriter(args["output"], fourcc, args["fps"],
			(w * 1, h * 1), True)
		# Construct the zeros array
		zeros = np.zeros((h, w), dtype="uint8")
	# Convert frame to a blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# Pass the blob through the network
	net.setInput(blob)
	# Obtain the detections and predictions
	detections = net.forward()

	# Loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# Extract the index of the class label from the `detections`
			idx = int(detections[0, 0, i, 1])
			# Compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


	# Construct the final output frame, storing the original frame
	output = np.zeros((h * 1, w * 1, 3), dtype="uint8")
	output[0:h, 0:w] = frame
			
	# Write the output frame to file
	writer.write(output)

	# Show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# If the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# Update the FPS counter
	fps.update()

# Stop the timer and display FPS information
# Do a bit of cleanup
print("[INFO] cleaning up...")
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
writer.release()
