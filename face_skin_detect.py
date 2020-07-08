import cv2
import sys
import numpy as np

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#Get camera settings
cam = input("[1] IP Camera\n[2] Webcam\nEnter your setting: ")
if cam == 1:
    ip_cam = raw_input("Enter IP Webcam Address (Ex: 192.168.0.8:9999): ")
    ip_cam = "http://" + ip_cam +"/video"
    video_capture = cv2.VideoCapture(ip_cam)
elif cam == 2:
    video_capture = cv2.VideoCapture(0)
else:
    print("Input error.")

#Lower and upper bound values taken from https://github.com/Jeanvit/PySkinDetection/blob/master/src/jeanCV.py
lower = np.array([0, 40, 0], dtype = "uint8")
upper = np.array([25, 255, 255], dtype = "uint8")

while True:
    #capture frames
    ret, frame = video_capture.read()
    frame = cv2.resize(frame,  (600, 400))
    
    #Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Binarize HSV frame to skin and non-skin
    skin_mask = cv2.inRange(hsv_frame, lower,upper)
    
    #erode and dilate on pixels to remove noice
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)
    
    #use gaussian blur to remove noise
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skin_mask)
    
    #detect face using skin detected
    faces = faceCascade.detectMultiScale(
        skin,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #draw bounding box on detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Displays 3 windows
    #skin_mask = binarized skin pixels
    #skin = pixels of skin_mask and frame together
    #frame = face with bounding box
    cv2.imshow('skin_mask',skin_mask)
    cv2.imshow('skin',skin)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()