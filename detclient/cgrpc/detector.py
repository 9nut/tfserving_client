import cv2
import grpc
import json
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class Detector:
    def __init__(self, hostport, network, labelspath):
        # Location of COCO dataset info; centernet_hourglass was trained on COCO 2017
        # dataset
        self.lpath = labelspath
        self.channel = grpc.insecure_channel(hostport)
        self.client = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.classnames = self.cococlasses(labelspath)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = network
        self.request.model_spec.signature_name = 'serving_default'

    # populate the COCO class names
    def cococlasses(self, annotpath):
        classnames = []
        with open(annotpath,'r') as cocoinfo:
            for line in cocoinfo:
                classnames.append(line.strip())

        print(classnames)
        return classnames

    def predict(self, image, label=None):
        # read image into numpy array
        img = cv2.imread(image).astype(np.uint8)
  
        # convert the image to tensor of shape:
        # batchsize ✕ height ✕ width ✕ colordepth
        tensor = tf.make_tensor_proto(img, shape=[1]+list(img.shape))
        self.request.inputs['input_tensor'].CopyFrom(tensor)
        reply = self.client.Predict(self.request, 30.0)
  
        # print(reply)

        classes = reply.outputs["detection_classes"].float_val
        scores = reply.outputs["detection_scores"].float_val
        boxes = reply.outputs['detection_boxes'].float_val

        # print(classes)
        # print(scores)
        print(boxes)

        detections = [{self.classnames[int(x)]:scores[i]} for i,x in enumerate(classes) if scores[i] >= 0.30]
    
        if label != None:
            detections = [x for x in detections if label in x]

        return detections
    
