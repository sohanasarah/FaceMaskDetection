# FaceMaskDetection
ML Course Project

----- Instructions---------

- First train the models using train_models.ipynb. It will create a feature extractor model named cnn.h5. It will also create classification models svm.pkl,knn.pkl and dt.pkl.
These models are necessary to run the mask_detection.py file.

- Next run the evaluation of models using evaluation.ipynb. At the last two lines of this notebook it runs mask_detection.py file using the best model found by performance analysis. We used SVM as the best model. A live camera will be opened to detect masks.

