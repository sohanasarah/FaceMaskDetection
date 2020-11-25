import cv2, os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from keras.models import load_model

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score

import matplotlib.pyplot as plt

import pickle


class DataPreProcessing:
    def __init__(self, data_path):
        self.data_path = data_path

    def pre_process_images(self):
        categories = os.listdir(self.data_path)
        # categories.remove('.DS_Store')
        labels = [i for i in range(len(categories))]
        label_dict = dict(zip(categories, labels))
        print(label_dict)
        img_size = 100
        data = []
        target = []
        for category in categories:
            folder_path = os.path.join(self.data_path, category)
            img_names = os.listdir(folder_path)
            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (img_size, img_size))
                    data.append(resized)
                    target.append(label_dict[category])
                except Exception as e:
                    print('Exception:', e)
        data = np.array(data) / 255.0
        data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
        target = np.array(target)
        # print(target)
        new_target = np_utils.to_categorical(target)
        np.save('images_preprocessed', data)
        np.save('targets_preprocessed', new_target)


class Cnn:
    def __init__(self):
        self.model = None

    def createModel(self, data):
        model = Sequential()
        model.add(Conv2D(200, (3, 3), input_shape=data.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(100, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu', name="name"))
        model.add(Dense(2, activation='sigmoid'))
        self.model = model

    def get_model(self):
        return self.model

    def compileFitModel(self, x, y):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(x, y, epochs = 7, validation_split = 0.2)
        return history

    def evaluateModel(self, x, y):
        return self.model.evaluate(x, y)

    def save_model(self, keras_model_path):
        model = self.get_model()
        model.save(keras_model_path)


class Svm:
    def __init__(self, feature_extractor, train_x, train_y, test_x, test_y):
        self.feature_extractor = feature_extractor
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def create_model(self, length):

        labels_x_train = np.zeros(length)

        for i in range(length):
            if self.train_y[i][0] == 1:
                labels_x_train[i] = 0
            else:
                labels_x_train[i] = 1

        labels_x_test = np.zeros(length)

        for i in range(length):
            if self.test_y[i][0] == 1:
                labels_x_test[i] = 0
            else:
                labels_x_test[i] = 1

        self.labels_x_test = labels_x_test

        clf = svm.SVC()
        clf.fit(np.array(self.feature_extractor(self.train_x[0:length], 0)), labels_x_train)
        self.clf = clf

    def predict(self, length):
        x_test_np = np.array(self.feature_extractor(self.test_x[0:length], 0))
        predicted = self.clf.predict(x_test_np)
        return predicted

    def get_model_accuracy(self, predictions):
        return accuracy_score(self.labels_x_test, predictions)

    def get_model(self):
        return self.clf

    def save_model(self, predictions):
        # Save model in the current working directory
        model  = self.get_model()
        y_test = self.labels_x_test
        tuple_objects = (model, y_test, predictions)
        pickle.dump(tuple_objects, open("svm.pkl", 'wb'))


class DecisionTrees:
    def __init__(self, feature_extractor, train_x, train_y, test_x, test_y):
        self.feature_extractor = feature_extractor
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def create_model(self, length):
        labels_x_train = np.zeros(length)
        for i in range(length):
            if self.train_y[i][0] == 1:
                labels_x_train[i] = 0
            else:
                labels_x_train[i] = 1
        labels_x_test = np.zeros(length)
        for i in range(length):
            if self.test_y[i][0] == 1:
                labels_x_test[i] = 0
            else:
                labels_x_test[i] = 1
        self.labels_x_test = labels_x_test

        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(np.array(self.feature_extractor(self.train_x[0:length], 0)), labels_x_train)
        self.tree = tree

    def predict(self, length):
        x_test_np = np.array(self.feature_extractor(self.test_x[0:length], 0))
        predicted = self.tree.predict(x_test_np)
        return predicted

    def get_model_accuracy(self, predictions):
        return accuracy_score(self.labels_x_test, predictions)
   

    def get_model(self):
        return self.tree

    def save_model(self, predictions):
        # Save model in the current working directory
        model  = self.get_model()
        y_test = self.labels_x_test
        tuple_objects = (model, y_test, predictions)
        pickle.dump(tuple_objects, open("dt.pkl", 'wb'))


class Knn:
    def __init__(self, feature_extractor, train_x, train_y, test_x, test_y, number_of_neighbors):
        self.feature_extractor = feature_extractor
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.number_of_neighbors = number_of_neighbors

    def create_model(self, length):
        labels_x_train = np.zeros(length)
        for i in range(length):
            if self.train_y[i][0] == 1:
                labels_x_train[i] = 0
            else:
                labels_x_train[i] = 1
        labels_x_test = np.zeros(length)
        for i in range(length):
            if self.test_y[i][0] == 1:
                labels_x_test[i] = 0
            else:
                labels_x_test[i] = 1
        self.labels_x_test = labels_x_test
        classifier = KNeighborsClassifier(n_neighbors=self.number_of_neighbors)
        classifier.fit(np.array(self.feature_extractor(self.train_x[0:length], 0)), labels_x_train)
        self.classifier = classifier

    def predict(self, length):
        x_test_np = np.array(self.feature_extractor(self.test_x[0:length], 0))
        return self.classifier.predict(x_test_np)

    def get_model_accuracy(self, predictions):
        return accuracy_score(self.labels_x_test, predictions)

    def get_model(self):
        return self.classifier

    def save_model(self, predictions):
        # Save model in the current working directory
        model  = self.get_model()
        y_test = self.labels_x_test
        tuple_objects = (model, y_test, predictions)
        pickle.dump(tuple_objects, open("knn.pkl", 'wb'))


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
    # preprocess images
    data__pre_processing = DataPreProcessing(dataPath)
    data__pre_processing.pre_process_images()

    # load data
    data = np.load('images_preprocessed.npy')
    target = np.load('targets_preprocessed.npy')

    # train-test split
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.1)
    
    return data, train_x, test_x, train_y, test_y


def train_cnn(data, train_x, test_x, train_y, test_y):
                     
    cnn = Cnn()
    cnn.createModel(data)
    history = cnn.compileFitModel(train_x, train_y)
    
    #save keras model 
    cnn.save_model("cnn.h5")

    plot_metrics(history, 'accuracy', 'CNN')
    print('accuracy cnn', cnn.evaluateModel(test_x, test_y))


def classification(feature_extractor, cnn_model, train_x, test_x, train_y, test_y):
                     
    print('Training SVM')
    svm = Svm(feature_extractor, train_x, train_y, test_x, test_y)
    svm.create_model(1000)
    svm_model = svm.get_model()
    svm_predictions = svm.predict(1000)
    
    accuracy_svm = svm.get_model_accuracy(svm_predictions)
    
    print('SVM accuracy:', accuracy_svm)

    svm.save_model(svm_predictions)
    
    
    print('Training Decision Tree')
    dt = DecisionTrees(feature_extractor, train_x, train_y, test_x, test_y)
    dt.create_model(1000)
    dt_predictions = dt.predict(1000)
    accuracy_dt = dt.get_model_accuracy(dt_predictions)
    
    print('DT accuracy', accuracy_dt)

    dt.save_model(dt_predictions)

    print('Training KNN')
    knn = Knn(feature_extractor, train_x, train_y, test_x, test_y, 9)
    knn.create_model(1000)
    knn_predictions = knn.predict(1000)
    accuracy_knn = knn.get_model_accuracy(knn_predictions)
    
    print('KNN accuracy', accuracy_knn)

    knn.save_model(knn_predictions)



def main(model= None):
    
    data, train_x, test_x, train_y, test_y = run_experiment('data')

    print(model)

    if model == 'cnn':
        
        train_cnn(data, train_x, test_x, train_y, test_y)

    elif model == 'classification':
        
        cnn = load_model("cnn.h5")

        feature_extractor = keras.Model(
            inputs = cnn.inputs,
            outputs= cnn.get_layer(name="name").output,
        )
        
        return classification(feature_extractor, cnn, train_x, test_x, train_y, test_y)

    elif model == None:
        
        train_cnn(data, train_x, test_x, train_y, test_y)
    
        cnn = load_model("cnn.h5")
        feature_extractor = keras.Model(
            inputs = cnn.inputs,
            outputs= cnn.get_layer(name="name").output,
        )
        return classification(feature_extractor, cnn, train_x, test_x, train_y, test_y)
