import dlib
import cv2
import numpy as np
from imutils import face_utils
def hog_face(path):
    rgb = cv2.imread(path)
    face_detect = dlib.get_frontal_face_detector()
    try:
        rects = face_detect(rgb, 1)
    except:
        print('Cannot find the image. Please try again.')
        return None
    crop_img=None
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        crop_img = rgb[y:y+h, x:x+w]
#        crop_img.append(rgb[y:y+h, x:x+w])
#        img = cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 255, 255), 3)
    return crop_img    
#a = hog_face('D:/CVProject/keras-vggface/CVFPgit/img/cat.jpg')
#import matplotlib.pyplot as plt 
#plt.imshow(a)
#for i in range(len(a)):
#    plt.figure(str(i))
#    plt.imshow(a[i])