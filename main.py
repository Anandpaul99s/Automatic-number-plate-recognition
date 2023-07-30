## coding the entire pipeline
import numpy as np
from ultralytics import YOLO
import cv2

import util
from util import get_car,read_license_plate , write_csv

from sort.sort import *

results = {}
mot_tracker = Sort() # in order to track all the vehicles
# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2,3,4,5,7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:

        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = [] ##saving bounding boxes of all vehicles
        for detection in detections.boxes.data.tolist():
            x1, y1 ,x2 ,y2 , score ,class_id = detection
            if int(class_id) in vehicles:
                detections_.append([ x1, y1 ,x2 ,y2 , score])

        # track vehicles ->tracking license plates in video frames
        track_ids = mot_tracker.update(np.asarray((detections_)))
        # contains all bounding boxes and bounding boxes related info
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to a car
            xcar1,ycar1,xcar2,ycar2,car_id = get_car(license_plate , track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2) , int(x1):int(x2), :]

                #process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64 , 255 , cv2.THRESH_BINARY_INV)
                ## TAKE PIXELS WHICH ARE LOWER THAN 64 AND IT TAKE UPTO 255

                 # read license plate
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car':{'bbox':[xcar1,ycar1,xcar2,ycar2]},
                                                    'license_plate': {'bbox':[x1,y1,x2,y2]
                                                    ,'text': license_plate_text,
                                                   'bbox_score': score,
                                                    'text_score': license_plate_text_score}}

# write reults
write_csv(results,'./test.csv')







