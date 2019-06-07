'''VGGFace models for Keras.

# Notes:
- Resnet50 and VGG16  are modified architectures from Keras Application folder. [Keras](https://keras.io)

- Squeeze and excitation block is taken from  [Squeeze and Excitation Networks in
 Keras](https://github.com/titu1994/keras-squeeze-excite-network) and modified.

'''



import numpy as np



def data_generator(data, targets, batch_size):
    batches = (len(data) + batch_size - 1)//batch_size
    while(True):
         for i in range(batches):
              X = data[i*batch_size : (i+1)*batch_size]
              Y = targets[i*batch_size : (i+1)*batch_size]
              yield (X, Y)
import pickle
def load_data():
    
    with open('D:/CVProject/keras-vggface/train_new.pickle', 'rb') as f:
        train_data = pickle.load(f)
        x_train = [x[0] for x in train_data]
        x_train = np.asarray(x_train).astype('float32')
        y_train = [int(x[1]) for x in train_data]
        y_train = np.asarray(y_train)
        y_ = np.zeros([len(y_train),100])
        y_[np.arange(len(y_train)),y_train] = 1

    x_train_1 = x_train#[:int(0.1*len(x_train))] 
    y_1 = y_#[:int(0.1*len(y_))] 
    return x_train_1,y_1       
def save_p(x_train_1,y_1):   
    with open('D:/CVProject/keras-vggface/trainx3.pickle', 'wb') as handle:
        pickle.dump(x_train_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('D:/CVProject/keras-vggface/trainy3.pickle', 'wb') as handle:
        pickle.dump(y_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load_p():              
    with open('D:/CVProject/keras-vggface/trainx3.pickle', 'rb') as f:
        x_train_1 = pickle.load(f)
    with open('D:/CVProject/keras-vggface/trainy3.pickle', 'rb') as f:
        y_1 = pickle.load(f)
    return x_train_1,y_1
#x_train_1,y_1=load_data()
##save_p(x_train_1,y_1)
#z = np.sum(y_1,axis = 0).reshape((20,5))
#z = np.sum(z,axis = 1)
#import matplotlib.pyplot as plt
#plt.plot(z)
#plt.show()
#save_p(x_train_1,y_1)
#y_1 = np.argmax(y_1,axis = 1)//5
#y_2 = np.zeros([len(y_1),20])
#y_2[np.arange(len(y_1)),y_1] = 1
#model = VGG16(include_top=True, weights='',
#          input_tensor=None, input_shape=[227,227,3],
#          pooling='max',
#          classes=20)
#adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
## checkpoint
#filepath="model/weights_best.hdf5"
#from keras.callbacks import ModelCheckpoint
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]
## Fit the model
#model.fit_generator(generator = data_generator(x_train_1, y_2, 32),
#                    steps_per_epoch = (x_train_1.shape[0] + 32 - 1) // 32,
#                    epochs = 5,
#                    verbose = 1,
#                    callbacks = callbacks_list
#                    
#)
##keras.backend.clear_session()

