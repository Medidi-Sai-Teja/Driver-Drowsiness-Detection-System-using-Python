from scipy.spatial import distance as dist
from imutils.video import VideoStream  
from imutils import face_utils
from threading import Thread
import pyttsx3 as sp
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

alarm_status=False
    
alarm_status2=False

def alarm(msg):
    
    eng=sp.init()
    # global alarm_status
    # global alarm_status2
    while alarm_status:
        print("eye alert is being executed")
        eng.say(msg)
        eng.runAndWait()
    if alarm_status2:
        print('yawn alert is being executed')
        eng.say(msg)
        eng.runAndWait()
    

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) 
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def mouth_aspect_ratio(shape):
    top_lip = shape[50:55]
    top_lip = np.concatenate((top_lip, shape[61:66]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68])) 

    A = dist.euclidean(top_lip[0], low_lip[4]) 
    B = dist.euclidean(top_lip[1], low_lip[3]) 
    C = dist.euclidean(top_lip[2], low_lip[2]) 
    D = dist.euclidean(top_lip[3], low_lip[1]) 
    E = dist.euclidean(top_lip[4], low_lip[0]) 
    F = dist.euclidean(shape[48], shape[54])
    
    mar = abs((A + B + C + D + E)/(2*F))
    return mar

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args()) 

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 25
YAWN_THRESH = 1.3

COUNTER = 0

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()

time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray)

    for rect in rects: 
        
        shape = predictor(gray, rect) 
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        mar = mouth_aspect_ratio(shape) 

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up!wake up!',))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "SLEEP ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        if (mar > YAWN_THRESH):
            cv2.putText(frame, "YAWN ALERT", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                
            if alarm_status2 == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take some fresh air',))
                t.deamon = True
                t.start()
           
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(mar), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()  
