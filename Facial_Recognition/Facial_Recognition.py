"""
Author: John Suchanek
Code Description: Facial Recognition with one reference image and Approximation
of the user's face from screen.
Date: Summer 2023
"""


import threading
import cv2
from deepface import DeepFace

#initialze webcam for capturing video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#set frame width and height for the captured video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#initialize variables
counter = 0
area = 0

face_match = False
john_match = False
other_match = False

#load reference images for face matching
reference_img = cv2.imread("reference.jpg")
other_img = cv2.imread("images.jpg")

#check if the image loaded successfully or not
if reference_img is not None:
    print("Image Loaded")
else:
    print("Image Did Not Load")

#function to check if a face in the current frame mathces the reference images
#uses DeepFace
def check_face(frame):
    #creating global boolean variables to be used in main code
    global face_match, john_match, other_match
    try:
        #verify if the face in the frame matches the reference image
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
            john_match = True
        elif DeepFace.verify(frame, other_img.copy())['verified']:
            face_match = True
            other_match = True
        else:
            face_match = False
    #handles errors during face verificaiton
    except ValueError:
        face_match = False
        other_match = False
        john_match = False

#function to draw a rectangle around detected faces
def draw_rectangle(frame, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#main loop for capturing and processing video frames
while True:
    #capture a frame from the webcam
    ret, frame = cap.read()
    #if frame was successfully captured
    if ret:
        #perform actions every 60 frames
        if counter % 60 == 0:
            try:
                #start new thread to check if the frame contains a matching frame
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        #if there is a face that matches
        if face_match:
            #display if it is John or someone else
            if john_match:
                cv2.putText(frame, "John", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            if other_match:
                cv2.putText(frame, "other", (240, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            #detect faces in frame and draw rectangles around them   
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                draw_rectangle(frame, x, y, w, h)
                area = w * h
            
            if area > (640 * 480 * 0.1):
                cv2.putText(frame, "FACE TOO CLOSE!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "UNKNOWN!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                draw_rectangle(frame, x, y, w, h)
                area = w * h
            
            if area > (640 * 480 * 0.1):
                cv2.putText(frame, "FACE TOO CLOSE!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
