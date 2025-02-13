# TCN-BiLSTM-CE

#### This github repository is used to reproduce the results of the model reconstructed for missing values in energy load data, TCN-BiLSTM-CE, including three parts:
#### load_data.py: Normalizing and Preprocessing the training data and model the training data and test data with sliding window
#### model_and_training.py: Modeling, compiling and training the TCN-BiLSTM-CE model and saving the trained model
#### test.py: Validating with test data

# Data

#### The data provided here for reproducing the results are electrical load data. Due to the data protection requirements of Green Fusion GmbH in Berlin, we can only provide the electrical load data here to make the results of the TCN-BiLSTM-CE reproducible.
#### URL for accessing the electricity dataset: https://github.com/buds-lab/building-data-genome-project-2/tree/master/data/meters/cleaned, here the author used the dataset called 'electricity_cleaned.csv'
#### Once you download the data from this github repo, put it under folder 'Data'. 'Data/Train_Data' is used for saving the training and testing data after pre-processing and normalizing with load_data.py
#### Selected buildings in the electricity load dataset:
#### Robin_public_Carolina, Bear_public_Rayna, Mouse_health_Ileana, Mouse_health_Estela, Hog_office_Denita, Hog_office_Napoleon, Wolf_retail_Toshia, Panther_retail_Kristina, Panther_parking_Lorriane, Panther_parking_Alaina, Panther_education_Teofila, Panther_education_Jerome

# Dependency

#### To avoid unnecessary trouble, the author recommends that you install the following version of the library:
#### tensorflow == 2.15.0
#### keras == 2.15.0
#### keras-tcn == 3.5.0
#### numpy == 1.26.4
#### pandas == 2.1.1
#### matplotlib == 3.6.3

