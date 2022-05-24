import numpy as np
import imutils
import cv2
import os
from imutils.video import VideoStream
haar_casecade = cv2.CascadeClassifier('haar_face.xml')
def read_path_img():
    path = r'student_datasets'
    return os.listdir(path)

#CATEGORIES = ['_19521501', '_19522430', '_19521799', '_19529999', '_19522347']
CATEGORIES = read_path_img()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
vs = VideoStream(src=0).start()
while True:
    image = vs.read()
    image = cv2.flip(image, 1)
    image = imutils.resize(image, width=800)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces_rect = haar_casecade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y: y+h, x: x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        if(confidence > 100):
            label = label + 1 if label + 1 < len(CATEGORIES) else label
        cv2.putText(image, str(CATEGORIES[label]), (x, y-3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=2)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv2.imshow('Detect Face', image)
    key = cv2.waitKey(1)
    if key == 27:
        break;
cv2.destroyAllWindows()