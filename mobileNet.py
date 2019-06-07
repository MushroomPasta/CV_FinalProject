import tensorflow as tf
import os
import time
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
os.chdir('D:/CVProject/keras-vggface/keras_vggface')
import load_pickle
import numpy as np
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
import cv2
from os import listdir
from os.path import isfile, join
def mini_batch(features,labels,mini_batch_size):
    l = features.shape[0]
    while True:
        for ndx in range(0, l, mini_batch_size):
            low = ndx
            high = min((ndx + mini_batch_size), l)
            yield shuffle(features[low:high,:], labels[low:high,:])
def build_model(target_size):
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_model = MobileNetV2(
        include_top=True,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='max')

    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
        
    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.5)(op)
    
    ##
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be 
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    ##
    output_tensor = Dense(10, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    learning_rate = 0.0001
    epoch = 10
    decay_rate = learning_rate / epoch
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model
def age_predict_2(pre1):
    index = np.argmax(pre1)
    prob1 = pre1[0,index]
    component1= prob1*10+index*10
    return component1
def load_resize(file):
    test_img = cv2.imread(file)
    
    img = cv2.resize(test_img,(224,224),interpolation=cv2.INTER_LANCZOS4).astype('float32')   
    img_t = img.reshape((1,224,224,3))
        
    return img_t,test_img
def resize(img):
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_LANCZOS4).astype('float32')   
    img_t = img.reshape((1,224,224,3))
    return img_t
def load_mdl(dirr):
    return load_model(dirr)
def predict_res(model, img):
    return model.predict(img)
def predict_age(img):
    print('start loading model...')
    model = load_mdl('D:/CVProject/keras-vggface/CVFPgit/weight/weights-improvement-05.hdf5')
    print('finish loading.')
    img = resize(img)
    print('start predicting...')
    pre = predict_res(model, img)
    age = age_predict_2(pre)
    print('Predicted age is: ',int(age),'.')
    



#if __name__ == "__main__":
    
#    x, y_1 = load_pickle.load_p()
#    y_1 = np.argmax(y_1,axis = 1)//10
#    y_2 = np.zeros([len(y_1),10])
#    y_2[np.arange(len(y_1)),y_1] = 1
#    X_train, X_test, y_train, y_test = train_test_split(x, y_2, test_size=0.05, random_state=42)
#    model = build_model(224)
#    model = load_model('weights-improvement-05.hdf5')
    
    
   
    
    
#    batch_size = 32
#    train_generator = mini_batch(X_train, y_train, batch_size)
#    # checkpoint
#    filepath="weights-improvement-{epoch:02d}.hdf5"
#    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, 
#                                 save_best_only=False, mode='auto',period = 5)
#    callbacks_list = [checkpoint]
#    
#    history = model.fit_generator(
#            generator=train_generator,
#            steps_per_epoch=x.shape[0]//batch_size,
#            verbose=1,
#            callbacks = callbacks_list,
#            validation_data = (X_test,y_test),
#            epochs=6)
#    model.save('model_k.h5')
    # Evaluate the model
#    (loss, accuracy) = model.evaluate(
#            x_test, 
#            y_test,
#            batch_size = 128, 
#            verbose = 1)
#    print(loss,accuracy)
#    pre = model.predict(X_test)
#    #confusion
#    from sklearn.metrics import confusion_matrix
#    matrix = confusion_matrix(y_test.argmax(axis=1), pre.argmax(axis=1))
#    total = np.sum(matrix)
#    tp_array = np.diag(matrix)+1e-10
#    detect = np.sum(matrix,axis = 0)+1e-10
#    true = np.sum(matrix,axis = 1)+1e-10
#    recall = tp_array/true
#    precision = tp_array/detect
#    tp = np.sum(np.diag(matrix))
#    accu = tp/total
#    #load predict image
#    
#    mypath = 'D:/CVProject/keras-vggface/'
#    test_files = [f for f in listdir(mypath+'test_face/') if isfile(join(mypath+'test_face/', f))]
#    age_lst = []
#    
#    for file in test_files:
#        img,test_img = load_resize(mypath+'test_face/'+file)
#        cv2.imwrite(mypath+'resized_test/'+file[:-3]+'jpg',img.reshape(224,224,3))
#        pre1 = model.predict(img)
#        age = int(age_predict_2(pre1))
#        age_lst.append(age)
#        image = Image.open(mypath+'resized_test/'+file[:-3]+'jpg')
#        draw = ImageDraw.Draw(image)
#        # desired size
# 
#        #font = ImageFont.truetype('Roboto-Bold.ttf', size=20)
# 
#        # starting position of the message
# 
#        (x, y) = (test_img.shape[0]//2, test_img.shape[1]//2)
#        message = 'age: ' + str(age)
#        color = 'rgb(211, 211, 211)' # black color
#        draw.text((x, y), message, fill=color)
#        image.save('D:/CVProject/keras-vggface/eval/'+'aged_'+file)
#        
#        
#        
#    
#    
#    
#    import matplotlib.pyplot as plt
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
#    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()