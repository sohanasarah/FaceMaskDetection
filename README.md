# FaceMaskDetection
ML Course Project

----- Instructions---------

- First train the models using train_models.ipynb. It will create a feature extractor model named cnn.h5. It will also create classification models svm.pkl,knn.pkl and dt.pkl.
These models are necessary to run the mask_detection.py file.

- evaluation.ipynb checks the performance of the saved models. At the last two lines of this notebook mask_detection.py file can be run using the choosen model. We haved SVM as as it gives the best performance. Running mask_detection.py will open camera and it will detect if someone is wearing masks or not. The model works well in good lighting. 

