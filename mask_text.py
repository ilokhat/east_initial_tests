import os

#from imutils.object_detection import non_max_suppression
import numpy as np
import cv2

# grabbed from https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
def decode_predictions(scores, geometry, min_confidence=0.1):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < min_confidence:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")



def load_pretrained_text_detector_model(path_to_east_model):
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(path_to_east_model)
    return net


def modif_filename(filename, motif="_masked"):
    basename, ext = os.path.splitext(filename)
    return f'{basename}{motif}{ext}'

def create_masked_image(image_path, east_net, padding=0.0, alpha= 0.75):
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # load the input image and grab the image dimensions
    image = cv2.imread(image_path)
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (origW, origH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward(layerNames)
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results_geoms = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX)
        startY = int(startY)
        endX = int(endX)
        endY = int(endY)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        results_geoms.append((startX, startY, endX, endY))

    for ((startX, startY, endX, endY)) in results_geoms:
        thickness = -1 # 2
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 0), thickness)
        
    # alpha Transparency factor.
    image_new = cv2.addWeighted(image, alpha, orig, 1 - alpha, 0)
    return image_new

def mask_all_images_in(images_dir, result_dir, east_net):
    onlyfiles = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    for f in onlyfiles:
        image_name = images_dir + "/" + f
        result_name = result_dir + "/" + modif_filename(f)
        image_new = create_masked_image(image_name, net)
        cv2.imwrite(result_name, image_new)
        print(image_name , "->", result_name)

EAST_MODEL  = './frozen_east_text_detection.pb'
net = load_pretrained_text_detector_model(EAST_MODEL)

IMG_DIR = './res'
IMG_RES = './masked1024'

# image_name = "./res/img_1_1.jpg"
# result_name = modif_filename(image_name)
# print(image_name, "->", modif_filename(image_name))
# image_new = create_masked_image(image_name, net)
# cv2.imwrite(result_name, image_new)

mask_all_images_in(IMG_DIR, IMG_RES, net)