import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #adding root dir to path env var.
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from utils.data import process_data
import pickle
import pandas as pd

def clean_data(data):
    """
    Cleans initial dataset.
    Removes whitespaces in both columns and all entries.
    Inputs: Initial raw dataset, expected as type dataframe.
    Returns: Cleaned dataset, of type dataframe.
    """
    assert((data.isna().sum()  == 0).all() == True) #ensure no missing values
    data.rename(str.strip, axis='columns', inplace=True) #remove all white-spaces from column names
    data = data.map(lambda x: x.strip() if type(x) == str else x) #removing all leading/trailing whitespaces from text/categorical data.
    return data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = AdaBoostClassifier(random_state=42)
    model = clf.fit(X_train, y_train)
    return model

def get_feature_names():
    """
    Get feature names for the dataset.
    Returns
    ---
    Lists of Categorical and numerical features
    """
    cat_features = [ 
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numerical_features = ["age", "education-num","capital-gain", "capital-loss", "hours-per-week"]
    return cat_features, numerical_features

def read_data():
    """
    Read raw census dataset
    Returns
    ---
    Raw census dataset, with type Dataframe
    """
    data_path = os.path.join(os.getcwd(),'utils',  'census.csv')
    data = pd.read_csv(data_path)
    return data

def save_fitted_data(model, encoder, label_binarizer):
    """
    Serialize and save off trained/fitted model and fitted encoders 
    Inputs
    ----
    model: trained model
    encoder: fitted one-hot-encoder
    label-binarizer: fitted label binarizer
    
    Returns
    ---
    None, files are saved.
    """
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)
    with open('label_binarizer.pkl', 'wb') as file:
        pickle.dump(label_binarizer, file)
    return

def load_fitted_data():
    """
    Load in and deserialize trained/fitted model and fitted encoders 
    Inputs
    ----
    None
    
    Returns
    ---
    model: trained model
    encoder: fitted one-hot-encoder
    label-binarizer: fitted label binarizer
    """
    with open(os.path.join('fitted_data', 'model.pkl'), 'rb') as file:
        model = pickle.load(file)
    with open(os.path.join('fitted_data','encoder.pkl'), 'rb') as file:
        encoder = pickle.load(file)
    with open(os.path.join('fitted_data','label_binarizer.pkl'), 'rb') as file:
        lb = pickle.load(file)
    return model, encoder, lb

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, F1, and accuracy.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    accuracy : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    accuracy = accuracy_score(y, preds)
    return precision, recall, fbeta, accuracy

def validate_data_slice_performance(data, model, feature, categorical_features, numerical_features, encoder, lb, logger):
    '''
    Validates the performance of the trained ML model on dataset slices of interest. 
    For categorical features, the slices evaluated are those with more than 1000 occurrences
    For each numerical feature, two slices are evaluated --  one includes data above the median value and one includes data below the median value.  
    
    Inputs 
    -----
    data: cleaned dataset from which to extract slices
    model: trained model
    feature: feature of interest to slice
    categorical_features: list of categorical features in dataset
    numerical_features: list of numerical features in dataset
    encoder: fitted OneHotEncoder
    lb: fitted Label-Binarizer
    logger: instance of logging module, used to output metrics for each of the data slices.

    Returns
    -----
    precisions: list of precisions for the evaluated slices
    accuracies: list of accuracies for the evaluated slices

    '''
    precisions = []
    accuracies = []

    #####Categorical Feature Data-Slice Validation
    if feature in categorical_features:
        feature_values = data[feature].value_counts().index.values
        feature_value_frequencies = data[feature].value_counts().values
        for feature_val, frequency in zip(feature_values, feature_value_frequencies):
            #Perform data split assessment if there are more than 1000 occurences of the feature value, or if slicing the education category.
            if frequency > 1000 or feature == 'education':
                #Extracting data subset with given feature value for categorical var of interest.
                data_subset = data[data[feature] == feature_val]
                assert(len(data_subset) == frequency) #ensuring extracted data subset has the proper frequency
                #Validating data slice performance
                precision_set, recall_set, fbeta_set, accuracy_set = eval_data_slice(data_subset, model, categorical_features, encoder, lb)
                logger.info(f"Feature: {feature}, Value: {feature_val}")
                logger.info(f"Precision: {precision_set:.2f}, Recall: {recall_set:.2f}, F1-Score: {fbeta_set:.2f}, Accuracy: {accuracy_set:.2f} ")
                precisions.append(precision_set); accuracies.append(accuracy_set)

    #####Numerical Feature Data-Slice Validation
    #Extracting medians for numerical feature, as this will inform our slices (> & < median value)
    num_feature_values = data.describe().loc['50%'].index.values
    num_feature_value_frequencies = data.describe().loc['50%'].values
    feat_map = dict(zip(num_feature_values, num_feature_value_frequencies))

    if feature in numerical_features:
        #Subset of data where the numerical feature is less than or equal to its median
        data_subset_1 = data[data[feature] <= feat_map[feature]] 
        # Validating data slice performance with metrics.
        precision_set, recall_set, fbeta_set, accuracy_set = eval_data_slice(data_subset_1, model, categorical_features, encoder, lb)
        logger.info(f"Feature: {feature}, Values <= {feat_map[feature]}")
        logger.info(f"Precision: {precision_set:.2f}, Recall: {recall_set:.2f}, F1-Score: {fbeta_set:.2f}, Accuracy: {accuracy_set:.2f} ")
        precisions.append(precision_set); accuracies.append(accuracy_set)
       
        #Subset of data where the numerical feature is greater than its median
        data_subset_2 = data[data[feature] > feat_map[feature]]
        # Validating data slice performance with metrics.
        precision_set, recall_set, fbeta_set, accuracy_set = eval_data_slice(data_subset_2, model, categorical_features, encoder, lb)
        logger.info(f"Feature: {feature}, Values > {feat_map[feature]}")
        logger.info(f"Precision: {precision_set:.2f}, Recall: {recall_set:.2f}, F1-Score: {fbeta_set:.2f}, Accuracy: {accuracy_set:.2f} ")
        precisions.append(precision_set); accuracies.append(accuracy_set)
    
    return precisions, accuracies

def eval_data_slice(data_subset, model, categorical_features, encoder, lb):
    '''
    Given a slice of cleaned data, this function uses the trained model to assess the classification metrics for the data slice.
    
    Inputs 
    -----
    data: cleaned clice/data subset
    model: trained model
    categorical_features: list of categorical features in dataset
    encoder: fitted OneHotEncoder
    lb: fitted Label-Binarizer
    
    Returns
    ----
    Metrics (precision, recall, f1-score, and accuracy) for the data slice of interest
    '''
    #splitting into features & values for data subset
    X_set, y_set, _, _ = process_data(data_subset, categorical_features,
                                            label="salary", training=False, encoder = encoder, lb = lb)       
    #performing inference with model, using X_set
    y_set_pred = model.predict(X_set)
    #Validating performance with metrics.
    precision_set, recall_set, fbeta_set, accuracy_set = compute_model_metrics(y_set, y_set_pred)
    
    return precision_set, recall_set, fbeta_set, accuracy_set

def inference(model, data, encoder, categorical_features):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Adaboost Classifier
        Trained machine learning model.
    data : Raw input, expected as dataframe
        Raw Data used for prediction.
    encoder: 
        Fitted one-hot-encoder, used for pre-processing raw inference dataset
    categorical features:
        Categorical features, used for one-hot-encoding.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    data_clean = clean_data(data)
    X_processed, _,_,_ = process_data(X = data_clean, categorical_features=categorical_features, training=False, encoder=encoder)
    preds = model.predict(X_processed)
    return preds

