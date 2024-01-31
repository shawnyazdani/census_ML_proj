import pandas as pd 
import os
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Script to train machine learning model.

data_path = os.path.join(os.getcwd(), '..',  'data' , 'census.csv')
data = pd.read_csv(data_path)
assert((data.isna().sum()  == 0).all() == True) #ensure no missing values
data.rename(str.strip, axis='columns', inplace=True) #remove all white-spaces from column names
data = data.map(lambda x: x.strip() if type(x) == str else x) #removing all leading/trailing whitespaces from text/categorical data.

# Splitting data, using train-test split instead of  K-fold cross validation 
train, test = train_test_split(data, test_size=0.20)

#Pre-processing  for train & test splits.
numerical_features = ["age", "education-num","capital-gain", "capital-loss", "hours-per-week"]
cat_features = [ #used for one-hot-encoding
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

#Fitted One-hot encoder and label binarizer are applied to test set. 
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
)

# Training and saving model -- saved using dvc.
model = train_model(X_train, y_train)
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

#Making predictions using trained model
Y_pred_train = inference(X_train)
Y_pred_test = inference(X_test)

#Assessing prediction metrics on both training & test set.
precision_train, recall_train, fbeta_train, accuracy_train = compute_model_metrics(y_train, Y_pred_train)
precision_test, recall_test, fbeta_test, accuracy_test = compute_model_metrics(y_test, Y_pred_test)
print(f"Test Precision: {precision_test:.2f}, Test Recall: {recall_test:.2f},\
 Test F1-Score: {fbeta_test:.2f}, Test Accuracy: {accuracy_test:.2f} ")
