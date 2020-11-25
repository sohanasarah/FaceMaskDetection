import cv2, os
import numpy as np
from tensorflow import keras
from keras.models import load_model
from tensorflow import keras

class CameraClass:
    def __init__(self):
        print("in camera class")

    def openCameraAndDetect(self, feature_extractor_model, predictor_model):
        source = cv2.VideoCapture(0)
        face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        labels_dict = {0: 'mask', 1: 'no_mask'}
        color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
        while (True):
            ret, img = source.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', gray)
            faces = face_clsfr.detectMultiScale(gray, 1.3, 5)
            label = None
            for (x, y, w, h) in faces:
                face_img = gray[y:y + w, x:x + w]
                resized = cv2.resize(face_img, (100, 100))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 100, 100, 1))
                if feature_extractor_model is not None:
                    feature_extractor = keras.Model(
                        inputs=feature_extractor_model.inputs,
                        outputs=feature_extractor_model.get_layer(name="name").output,
                    )
                    exTrain = feature_extractor(reshaped, 0)
                    x_train = np.array(exTrain)
                    result = predictor_model.predict(x_train)
                    label = result[0]
                else:
                    label = np.argmax(predictor_model.predict(reshaped), axis=1)[0]
                    print(label)
                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
                cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow('LIVE', img)
            key = cv2.waitKey(1)
            if (key == 27):
                break
        cv2.destroyAllWindows()
        source.release()

        
def main(model_type, model = None):
    cnn = load_model("cnn.h5")
    
    cameraClass = CameraClass()

    if model_type == "deep_learning":
        cameraClass.openCameraAndDetect(None, cnn)
    else:
        cameraClass.openCameraAndDetect(cnn, model)