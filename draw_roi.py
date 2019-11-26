import cv2
import numpy as np 

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def begueradj_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, points

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
        points = []

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(255,0,0),3)
                current_former_x = former_x
                current_former_y = former_y
                points.append([current_former_x, current_former_y])
            else:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,255,0),3)
                current_former_x = former_x
                current_former_y = former_y
                points.append([current_former_x, current_former_y])

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(255,0,0),3)
            current_former_x = former_x
            current_former_y = former_y
            points.append([current_former_x, current_former_y])
            count = np.asarray(points).reshape((-1,1,2)).astype(np.int32)
            cv2.fillPoly(mask, pts =[count], color=(255,255,255))
        else: 
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,255,0),3)
            current_former_x = former_x
            current_former_y = former_y
            points.append([current_former_x, current_former_y])
            count = np.asarray(points).reshape((-1,1,2)).astype(np.int32)
            cv2.fillPoly(mask, pts =[count], color=(0,0,0))

    return former_x,former_y    

im = cv2.imread("input/backgroud.jpg")

height=im.shape[0] #Input image size
width=im.shape[1]
mask = np.zeros((height, width), np.float64) #Belief grid

cv2.namedWindow("Drawer", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Drawer', (600,600))
cv2.setMouseCallback('Drawer',begueradj_draw)
while(1):
    cv2.imshow('Drawer',im)
    cv2.imshow('Mask',mask)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27: #ESC
        cv2.imwrite("images/circles_{}.jpg".format(2), im)
        cv2.imwrite("images/mask_{}.jpg".format(2), mask)
        break
cv2.destroyAllWindows()