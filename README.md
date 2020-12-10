# FaceMaskDetection
ML Course Project

--- Folder Structure -----

--data
--models.py
--mask_detection.py
--train_models.ipynb
--evaluation.ipynb

----- Instructions---------

- Link to the dataset is given data.txt. Please download the dataset before training the model. The data folder should be on the same folder with all the .py and .ipynb files.

- Train the models using train_models.ipynb. It will create a feature extractor model named cnn.h5. It will also create classification models svm.pkl,knn.pkl and dt.pkl.
These models are necessary to run the mask_detection.py file.

- evaluation.ipynb checks the performance of the saved models. At the last two lines of this notebook, mask_detection.py file can be run using the chosen model. We have SVM as the chosen model as it gives the best performance among all the classification models. Running mask_detection.py will open camera and it will detect if someone is wearing mask or not.

