import numpy as np
from pyo import *
import cv2
import imutils

# some initial state
camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
scale_factor = 1200
gray = cv2.COLOR_BGR2GRAY
frame = None

# pyo Setup
pyoServer = Server().boot()
pyoServer.start()
pyoServer.gui(locals())

mod = Sine(freq=4, mul=50)
sin = Sine(freq=mod + 880, mul = 0.2).out()

# capture the first frame for motion tracking reference
firstFrame = None
(_, firstFrame) = camera.read()
firstFrame = imutils.resize(firstFrame, width=600)
firstFrame = cv2.cvtColor(firstFrame, gray)

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    
    frame = imutils.resize(frame, width=600)
    frame = cv2.cvtColor(frame, gray)

    frameDelta = cv2.absdiff(firstFrame, frame)
    frame = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    frame_fft = np.fft.fft2(frame)
    
    mean = []
    for i in frame_fft[0]:
        mean.append(i)

    mean = np.mean(mean)

    scaled_mean = np.absolute(mean / scale_factor)
    pyoUpdate(scaled_mean)
    
    cv2.putText(frame, str(scaled_mean), (10,500), font, 1, (255,255,255), 2)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
pyoServer.stop()
