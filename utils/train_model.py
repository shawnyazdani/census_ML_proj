import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #adding root dir to path env var.
from sklearn.model_selection import train_test_split
from utils.data import process_data
from utils.model import train_model, compute_model_metrics, clean_data, save_fitted_data, get_feature_names, read_data

# Script to train machine learning model.
data = read_data()
#cleaning dataset
data_clean = clean_data(data)
# Splitting data, using train-test split instead of  K-fold cross validation 
train, test = train_test_split(data_clean, test_size=0.20)

#Pre-processing  for train & test splits.
cat_features, numerical_features = get_feature_names() #used for one-hot-encoding
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

#Fitted One-hot encoder and label binarizer are applied to test set. 
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
)

# Training and saving model -- saved using dvc.
model = train_model(X_train, y_train)

#Saving off fitted model and encoders.
save_fitted_data(model, encoder, lb)
                 
#Making predictions using trained model
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

#Assessing prediction metrics on both training & test set.
precision_train, recall_train, fbeta_train, accuracy_train = compute_model_metrics(y_train, Y_pred_train)
precision_test, recall_test, fbeta_test, accuracy_test = compute_model_metrics(y_test, Y_pred_test)
print(f"Test Precision: {precision_test:.2f}, Test Recall: {recall_test:.2f},\
 Test F1-Score: {fbeta_test:.2f}, Test Accuracy: {accuracy_test:.2f} ")
