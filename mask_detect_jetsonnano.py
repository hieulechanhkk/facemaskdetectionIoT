# import numpy as np
# import time
import cv2
from imutils.video import VideoStream
import imutils
# from PIL import Image
import numpy as np
import time
from tensorflow.keras.preprocessing.image import img_to_array
# from keras.utils.np_utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from datetime import datetime
import serial

import paho.mqtt.client as paho


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
broker="broker.hivemq.com"
port=1883
def on_publish(client,userdata,result):             #create function for callback
    print("data published \n")
    pass



# arduino = SerialObject('COM6', 115200)


def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

init_firebase_authorize()
prototxtPath = r'face_detect/deploy.prototxt.txt'
weightPath = r'face_detect/res10_300x300_ssd_iter_140000.caffemodel'


model = load_model("mask_detector.model")
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightPath)

vs = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
flagNMask = 0
flagMask = 0

client1 = paho.Client("control1")  # create client object
client1.on_publish = on_publish  # assign function to callback
client1.connect(broker, port)
client1.loop_start()

prev_frame_time = 0

new_frame_time = 0
while True:
    try:
        ret, image = vs.read()
        image = cv2.flip(image, 1)
        image = imutils.resize(image, width=800)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        faces = []
        preds = []
        locs = []
        fps = int(fps)

        fps = str(fps)
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

        if (len(faces) > 0):
            faces = np.array(faces, dtype="float32")
            preds = model.predict(faces, batch_size=32)
        for (loc, pred) in zip(locs, preds):
            (x0, y0, x1, y1) = loc
            (with_mask, without_mask) = pred

            label = "Mask" if with_mask > without_mask else "Without Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(image, (x0, y0 - 23), (x1, y0 - 3), color, -2)
            cv2.putText(image, label, (x0 + 5, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

            if with_mask > without_mask:
                if (flagMask == 0):
                    print("Send Data Mask")  # $1
                    flagMask = 1
                    flagNMask = 0
                    rett= client1.publish("project/mask", "1")
                    # arduino.sendData([1])
            else:
                if (flagNMask == 0):
                    print("Send Data No Mask")  # $0
                    flagMask = 0
                    flagNMask = 1
                    rett = client1.publish("project/mask", "0")

                    dt_string = get_date_time()
                    cv2.imwrite(f'Photos_capture/Unknown/Unknown_{dt_string}.jpg', image)
                    Upload_img_to_firebase(f'Photos_capture/Unknown/Unknown_{dt_string}.jpg',
                                           "Unknown", dt_string)
                    preLabel = label
                    # arduino.sendData([0])


        cv2.imshow('Image', image)
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):
            break
    except:
        ret, image = vs.read()
        image = cv2.flip(image, 1)
        image = imutils.resize(image, width=800)
        cv2.imshow('Image', image)
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):
            break
client1.disconnect()
cv2.destroyAllWindows()
vs.stop()