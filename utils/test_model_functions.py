import pandas as pd
import logging
from utils.model import validate_data_slice_performance, compute_model_metrics, inference, load_fitted_data, get_feature_names, read_data, clean_data

data = read_data()
data_clean = clean_data(data)
cat_features, numerical_features = get_feature_names() #used for data slicing
model, encoder, lb = load_fitted_data() 

def configure_logging(log_outputs = False):
    '''
    Configures instance of logger. Logs stdout to a text file if desired (e.g. to view data slice metrics).
    Inputs: Bool indicating whether to log outputs to file (true) or not (false, default).
    Returns: Logger instance
    '''
    if log_outputs:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', filemode='a')
        file_handler = logging.FileHandler('slice_output.txt')
        file_handler.setLevel(logging.INFO)
        logger = logging.getLogger()
        logger.addHandler(file_handler)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
        logger = logging.getLogger()
    return logger

logger = configure_logging()

precision_threshold = 0.55
accuracy_threshold = 0.7
tolerance = 0.0001

def test_categorical_slice_precisions_accuracies():
    """ Test to see if a data slice of each feature of our dataset produces proper accuracies and precisions"""
    for feature in cat_features:
        _, accuracies = validate_data_slice_performance(data_clean, model, feature, cat_features, numerical_features, encoder, lb, logger)
        assert(all([accuracy >= accuracy_threshold for accuracy in accuracies])),  f"Failed for {feature}"
        

def test_numerical_slice_precisions_accuracies():
    """ Test to see if a data slice of each feature of our dataset produces proper accuracies and precisions"""
    for feature in numerical_features:
        precisions, accuracies = validate_data_slice_performance(data_clean, model, feature, cat_features, numerical_features, encoder, lb, logger)
        assert(all([precision >= precision_threshold for precision in precisions])), f"Failed for {feature}" 
        assert(all([accuracy >= accuracy_threshold for accuracy in accuracies])),  f"Failed for {feature}"

def test_metrics_perfect_classification():
    """ Ensure metrics are calculated properly, for a case with perfect classification"""
    y_true = [1,0,1,0]
    y_pred = [1,0,1,0]
    precision, recall, fbeta, accuracy = compute_model_metrics(y_true, y_pred)
    for val in [precision, recall, fbeta, accuracy]:
        assert(abs(val-1) < tolerance)

def test_metrics_improper_classification():
    """ Ensure metrics are calculated properly, for a case with improper classification"""
    y_true = [1,0,1,0]
    y_pred = [0,1,0,1]
    precision, recall, fbeta, accuracy = compute_model_metrics(y_true, y_pred)
    for val in [precision, recall, accuracy]:
        assert(abs(val-0) < tolerance)

def test_inference_output_shape():
    """ Ensure that inference produces the same output shape as the input shape."""
    # Inference with Multi-sample entry case, excluding labels from input raw dataset 
    multi_sample = data.iloc[42:48,:-1]
    multi_predictions = inference(model, multi_sample, encoder, cat_features)
    assert( len(multi_predictions) == len(multi_sample) )
    print(f"Multi-Sample Inference Predictions: {multi_predictions}")

    #Inference with Single-sample entry case, excluding labels from input raw dataset
    single_sample = pd.DataFrame(data.iloc[65,:-1]).transpose()
    single_prediction = inference(model, single_sample, encoder, cat_features)
    assert( len(single_prediction) == len(single_sample) )
    print(f"Single-Sample Inference Predictions: {single_prediction}")
