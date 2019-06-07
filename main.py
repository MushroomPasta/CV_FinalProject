import matplotlib.pyplot as plt 
from HOG import hog_face
import mobileNet as k

if __name__ == '__main__':
    print('start loading model...(this may take a minute)')
    model = k.load_mdl('weight/weights-improvement-05.hdf5')
    print('finish loading.')
    #'D:/CVProject/keras-vggface/CVFPgit/img/zj.jpg'
    while True:
        path = input("Please enter your full path of one image: ")
        img = hog_face(path)
        if img is None:
            print('Face not detected, please try another image')
        else:
            img = k.resize(img)
            print('start predicting...')
            pre = k.predict_res(model, img)
            age = k.age_predict_2(pre)
            print('Predicted age is: ',int(age),'.')
        query = input("Do you want to continue with another image? (Y/N)")
        if query[0].upper() == 'N':
            print("Thank you for trying out our application, Goodbye!")
            break