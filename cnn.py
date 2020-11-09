import cv2,os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import svm
from sklearn.metrics import accuracy_score
class DataPreProcessing:
    def __init__(self, data_path):
        self.data_path=data_path
    def pre_process_images(self):
        categories=os.listdir(self.data_path)
        #categories.remove('.DS_Store')
        labels=[i for i in range(len(categories))]
        label_dict=dict(zip(categories,labels))
        print(label_dict)
        img_size=100
        data=[]
        target=[]
        for category in categories:
            folder_path=os.path.join(self.data_path,category)
            img_names=os.listdir(folder_path)
            for img_name in img_names:
                img_path=os.path.join(folder_path,img_name)
                img=cv2.imread(img_path)
                try:
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    resized=cv2.resize(gray,(img_size,img_size))
                    data.append(resized)
                    target.append(label_dict[category])
                except Exception as e:
                    print('Exception:',e)
        data=np.array(data)/255.0
        data=np.reshape(data,(data.shape[0],img_size,img_size,1))
        target=np.array(target)
        print(target)
        new_target=np_utils.to_categorical(target)
        np.save('images_preprocessed',data)
        np.save('targets_preprocessed',new_target)
class Cnn:
    def __init__(self):
        self.model = None
    def createModel(self, data):
        model=Sequential()
        model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(100,(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(50,activation='relu', name="name"))
        model.add(Dense(2,activation='sigmoid'))
        self.model = model
    def getModel(self):
        return self.model
    def compileFitModel(self, x, y):
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history=self.model.fit(x,y,epochs=1,validation_split=0.2)
        return history
    def evaluateModel(self, x, y):
        return self.model.evaluate(x, y)
class Svm:
    def __init__(self, feature_extractor, train_x, train_y, test_x, test_y):
        self.feature_extractor = feature_extractor
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
    def create_model(self, length):
        labels_x_train= np.zeros(length)
        for i in range(length):
            if self.train_y[i][0] == 1:
                labels_x_train[i] = 0
            else:
                labels_x_train[i] = 1
        labels_x_test= np.zeros(length)
        for i in range(length):
            if self.test_y[i][0] == 1:
                labels_x_test[i] = 0
            else:
                labels_x_test[i] = 1
        self.labels_x_test = labels_x_test
        clf = svm.SVC()
        clf.fit(np.array(self.feature_extractor(self.train_x[0:length],0)), labels_x_train)
        self.clf = clf
    def predict(self, length):
        x_test_np = np.array(self.feature_extractor(self.test_x[0:length],0))
        predicted = self.clf.predict(x_test_np)
        return predicted
    def get_model_accuracy(self, predictions):
        return accuracy_score(self.labels_x_test, predictions)
    def get_model(self):
        return self.clf
class CameraClass:
    def __init__(self):
        print("in camera class")
    def openCameraAndDetect(self, feature_extractor_model, predictor_model):
        source      = cv2.VideoCapture(0)
        face_clsfr  = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        labels_dict = {0:'mask',1:'no_mask'}
        color_dict  = {0:(0,255,0),1:(0,0,255)}
        while(True):
            ret,img = source.read()
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('frame', gray)
            faces = face_clsfr.detectMultiScale(gray, 1.3, 5)
            label = None
            for (x,y,w,h) in faces:
                face_img = gray[ y:y+w, x:x+w ]
                resized = cv2.resize(face_img,(100,100))
                normalized = resized/255.0
                reshaped = np.reshape(normalized,(1,100,100,1))
                if feature_extractor_model is not None:
                    feature_extractor = keras.Model(
                         inputs  = feature_extractor_model.inputs,
                         outputs = feature_extractor_model.get_layer(name="name").output,
                     )
                    exTrain = feature_extractor(reshaped, 0)
                    x_train = np.array(exTrain)
                    result = predictor_model.predict(x_train)
                    label = result[0]
                else:
                    label = np.argmax(predictor_model.predict(reshaped),axis=1)[0]
                    print(label)
                cv2.rectangle(img,(x,y),(x+w,y+h), color_dict[label], 2)
                cv2.rectangle(img,(x,y-40),(x+w,y), color_dict[label], -1)
                cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow('LIVE',img)
            key=cv2.waitKey(1)
            if(key==27):
                break
        cv2.destroyAllWindows()
        source.release()
def plot_metrics(history, metric_name, model_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, len(history.history[metric_name]) + 1)
    plt.plot(e, metric, '-bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, '-go', label='Validation ' + metric_name)
    plt.title('Training data ' + metric_name + ' & ' + 'validation data ' + metric_name + ' using ' + model_name)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
def run_experiment(dataPath):
    #preprocess images
    data__pre_processing = DataPreProcessing(dataPath)
    data__pre_processing.pre_process_images()
    #load data
    data=np.load('images_preprocessed.npy')
    target=np.load('targets_preprocessed.npy')
    cnn = Cnn()
    cnn.createModel(data)
    train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.1)
    history = cnn.compileFitModel(train_x, train_y)
    plot_metrics(history, 'accuracy', 'CNN')
    print(cnn.evaluateModel(test_x, test_y))
    feature_extractor = keras.Model(
        inputs=cnn.getModel().inputs,
        outputs=cnn.getModel().get_layer(name="name").output,
    )
    cameraClass = CameraClass()
    model_used = "svm"
    if model_used == "deep_learning":
    #Deep Learning
        cameraClass.openCameraAndDetect(None, cnn.getModel())
    elif model_used == "svm":
    #SVM
        svm = Svm(feature_extractor, train_x, train_y, test_x, test_y)
        svm.create_model(1000)
        svm_predictions = svm.predict(1000)
        accuracy = svm.get_model_accuracy(svm_predictions)
        cameraClass.openCameraAndDetect(cnn.getModel(), svm.get_model())
    #decision Tress
    #K Nearest Neighbor
def main():
    return run_experiment('data')