import cv2
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
def face_extractor(file):
    img  = cv2.imread(file)
    cascPath = "D:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('File not found. Please try again')
        return None

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100,100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    crop_img = None
    for (x, y, w, h) in faces:    
        crop_img = img[y:y+h, x:x+w]
    
    return crop_img
#crop_img = face_extractor('D:/CVProject/keras-vggface/CVFPgit/img/z.jpg')
#plt.imshow(crop_img)
#list_1 = ['00','01','02','03','04','05','06','07','08','09']
#list_2 = [str(x) for x in range(10,100)]
#lst = list_1+list_2
##if not os.path.isdir('D:/CVProject/face'):
##    os.makedirs('D:/CVProject/face')
##for name in lst:
##    if not os.path.isdir('D:/CVProject/face/'+name):
##        os.makedirs('D:/CVProject/face/'+name)
#base_dirs = 'D:/wiki-fullimage/wiki/'
#aim_dirs = 'D:/CVProject/face/'
#for name in tqdm(lst):
#    onlyfiles = [f for f in listdir(base_dirs+name) if isfile(join(base_dirs+name, f))]
#    for files in onlyfiles:
#        if not os.path.isfile(aim_dirs+name+'/'+files):
#            img  = cv2.imread(base_dirs+name+'/'+files)
#            crop_img = face_extractor(img,base_dirs+name+'/'+files)
#            try:
#                cv2.imwrite(aim_dirs+name+'/'+files,crop_img)
#            except NameError:
#                continue
        
        
        



