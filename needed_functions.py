import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

import imutils
import imageio
import imgaug.augmenters as iaa
import imgaug as ia
from yolo import YOLO
import argparse
import urllib
from tensorflow.keras.models import load_model

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
 
        
    for (x, y, w, h) in faces:
         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
         
    if len(faces) == 1:
        return True
    else:
        return False
    
def prepare(new_frame):
    IMG_SIZE = 227
    
    flip = iaa.Fliplr(1.0)
    zoom = iaa.Affine(scale=1)
    random_brightness = iaa.Multiply((1, 1.2))
    rotate = iaa.Affine(rotate=(-20, 20))
    
    
    #orig_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    image_aug = cv2.resize(new_frame, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_CUBIC)              
    image_aug = flip(image = image_aug)
    image_aug = random_brightness(image = image_aug)
    image_aug = zoom(image = image_aug)
    image_aug = rotate(image = image_aug)
    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
    new_array = np.array(image_aug)
    print(new_array.shape)
    new_array = new_array.reshape(-1, IMG_SIZE * IMG_SIZE * 3)
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    print(new_array.shape)
    new_array = new_array / 255.0
    return new_array



def detect_hands(frame):
    
    
    model = load_model('D:/Liveliness_test/hand_detection/models/AlexNetGesturesRecognizer.h5')
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn / v4-tiny')
    ap.add_argument('-d', '--device', default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=416, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
    ap.add_argument('-nh', '--hands', default=1, help='Total number of hands to be detected per frame (-1 for all)')
    args = ap.parse_args()

    if args.network == "normal":
        print("loading yolo...")
        yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
    
    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)
    
    
    width, height, inference_time, results = yolo.inference(frame)

    # display fps
    #cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)

    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    if args.hands != -1:
        and_count = int(args.hands)

    # display hands
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    
        crop_img = frame[y:y+h, x:x+w]
        #crop_img = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        #crop_image = cv2.resize(crop_img, (200, 200))
        
        #min_HSV = np.array([0, 58, 30], dtype = "uint8")
        #max_HSV = np.array([33, 255, 255], dtype = "uint8") 
        
        #min_HSV = np.array([0, 10, 60], dtype = "uint8") 
        #max_HSV = np.array([20, 150, 255], dtype = "uint8")
        
        min_HSV = np.array([0, 48, 80], dtype = "uint8")
        max_HSV = np.array([20, 255, 255], dtype = "uint8") 
    
        imageHSV = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    
        prediction = model.predict(prepare(skinRegionHSV))
        
        for i in range(0, len(np.round(prediction[0]))):
            
            if i==0 and np.round(prediction[0][i])==1.0:
               # print("FIVE")
                msg = "FIVE"
                return 0
            if i==1 and np.round(prediction[0][i])==1.0:
                #print("THUMBSUP")
                msg = "THUMBSUP"
                return 1
            if i==2 and np.round(prediction[0][i])==1.0: 
                msg = "TWO"
                #print("TWO")
                return 2
            if i==3 and np.round(prediction[0][i])==1.0:
                msg = "YO"
                #print("YO")
                return 3
            
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image