import numpy as np
import time
import cv2
from imutils.video import VideoStream
from datetime import datetime
import imutils
from PIL import Image
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model

from cvzone.SerialModule import SerialObject

from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


def init_firebase_authorize():
    cred = credentials.Certificate("firebase/aiotformaskdetection-firebase-adminsdk-s79ro-5f7cc5b7d1.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'aiotformaskdetection.appspot.com'})

def Upload_img_to_firebase(img_path, student_code, time):
    bucket = storage.bucket()
    filename = student_code + "_" + time + '.jpg'
    blob = bucket.blob(f'{student_code}/{filename}')
    blob.upload_from_filename(img_path)

def get_date_time():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%Hh%Mm%Ss")
    return dt_string

preLabel = -1
def face_profile(faces_rett, colorr, gray, boxx, imagee):
    global preLabel
    for (xx, yy, ww, hh) in faces_rett:
        faces_roi = gray[yy: yy+hh, xx: xx+ww]
        label, confidence = face_recognizer.predict(faces_roi)

        if (confidence > 100):
            label = label + 1 if label + 1 < len(CATEGORIES) else label

        (xx, yy, ww, hh) = boxx
        student_codes = str(CATEGORIES[label]).split("_")
        student_code = student_codes[1]
        cv2.putText(imagee, student_code, (xx, yy - 3), cv2.FONT_HERSHEY_COMPLEX, 0.5, colorr, thickness=1)
        if preLabel == -1 or preLabel != label:
            dt_string = get_date_time()
            path_length = str(len(os.listdir(r'Photos_capture')) + 1)
            try:
                os.mkdir(f'Photos_capture/{student_code}')
            except:
                continue
            cv2.imwrite(f'Photos_capture/{student_code}/{student_code}_{dt_string}.jpg', imagee)
            Upload_img_to_firebase(f'Photos_capture/{student_code}/{student_code}_{dt_string}.jpg', student_code, dt_string)
            preLabel = label


init_firebase_authorize()

arduino = SerialObject('COM6', 115200)

prototxtPath = r'face_detect/deploy.prototxt.txt'
weightPath = r'face_detect/res10_300x300_ssd_iter_140000.caffemodel'
#Load model recognize
haar_cascade = cv2.CascadeClassifier(r'face_recognize/haar_face.xml')
CATEGORIES = ['_19521501', '_19522430', '_19521799', '_19529999', '_19522347']
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognize/face_trained.yml')


model = load_model("mask_detector.model")
faceNet = cv2.dnn.readNet(prototxtPath, weightPath)

vs = VideoStream(src=0).start()
flagNMask = 0
flagMask = 0

while True:
    image = vs.read()
    image = cv2.flip(image, 1)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    preds = []
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)

        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    if(len(faces) > 0):
        faces = np.array(faces, dtype="float32")
        preds = model.predict(faces, batch_size=32)
    for (loc, pred) in zip(locs, preds):

        (x0, y0, x1, y1) = loc
        (with_mask, without_mask) = pred

        label = "Mask" if with_mask > without_mask else "Without Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(image, (x0, y0 - 23), (x1, y0 - 3), color, -2)
        cv2.putText(image, label, (x0 + 5, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        student_code = 0
        if label == 'Without Mask':
            box = (x0, y0 - 25, x1, y1)
            face_profile(faces_rect, (255, 255, 255), gray, box,image)
        if with_mask > without_mask:
            if (flagMask == 0):
                print("Send Data Mask")  # $1
                flagMask = 1
                flagNMask = 0
                arduino.sendData([1])
        else:
            if (flagNMask == 0):
                print("Send Data No Mask")  # $0
                flagMask = 0
                flagNMask = 1
                arduino.sendData([0])


    cv2.imshow('Detect Mask & Arduino', image)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
vs.stop()