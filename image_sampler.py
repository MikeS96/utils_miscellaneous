'''
Using OpenCV takes a mp4 video and produces a number of images.

Requirements
----
You require OpenCV 3.2 to be installed.

Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py

Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('Training.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 404
fcounter = 0
success = True #Flag to extract frames ONLY when the video is running

while(success):
    # Capture frame-by-frame
    success, frame = cap.read()

    # Saves image of the current frame in jpg file every 200 frames
    #For Training video condition > 45
    #For Test video condition > 200
    if fcounter > 45:
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        #Resets the frames counter
        fcounter = 0
        # To stop duplicate images
        currentFrame += 1

    fcounter += 1

    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()