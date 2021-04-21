import argparse
import cv2
import json
import requests

# REST API URL:
# Example: for local tensorflow-serving using centernet_hourglass network
# The metadata URL is:
#   http://localhost:8501/v1/models/centernet_hourglass_512x512_kpts/metadata
#
# Canonical form of the URI is:
# http://${hostport}/v1/models/${network}:${method}
# where method is "predict"
# example:
#   http://localhost:8501/v1/models/centernet_hourglass_512x512_kpts:predict

class Detector:
    def __init__(self, hostport, network, labelspath):
        self.lpath = labelspath
        self.URI = 'http://'+hostport+'/v1/models/'+network
        self.classnames = self.cococlasses(labelspath)

    # populate the COCO class names
    def cococlasses(self, annotpath):
        classnames = []
        with open(annotpath,'r') as cocoinfo:
            for line in cocoinfo:
                classnames.append(line.strip())

        print(classnames)
        return classnames

    # Detect objects using the REST API
    # returns a list of detected objects and only
    # includes the labels that match "label" if it is
    # provided
    def predict(self, image, label=None):
        headers = {"content-type": "application/json"}

        # Check the "metadata" URL for request and response formats

        image_content = cv2.imread(image,1).astype('uint8').tolist()
        resturi = self.URI+':predict'
        # print(resturi)

        reply = requests.post(resturi, data=json.dumps({"inputs" : [ image_content ]}), headers = headers) 
        # print(reply.text)
        jrep = reply.json()

        classes = jrep["outputs"]["detection_classes"][0]
        scores = jrep["outputs"]["detection_scores"][0]
        boxes = jrep["outputs"]["detection_boxes"][0]

        print(boxes)

        detections = [{self.classnames[int(x)]:scores[i]} for i,x in enumerate(classes) if scores[i] >= 0.30 ]
    
        if label != None:
            detections = [x for x in detections if label in x]

        return detections
