import pandas as pd
from utils.train_model import data, data_clean, cat_features, numerical_features, model, encoder, lb, X_train
from utils.model import validate_data_slice_performance, compute_model_metrics, inference

precision_threshold = 0.55
accuracy_threshold = 0.7
tolerance = 0.0001

def test_categorical_slice_precisions_accuracies():
    """ Test to see if a data slice of each feature of our dataset produces proper accuracies and precisions"""
    for feature in cat_features:
        precisions, accuracies = validate_data_slice_performance(data_clean, model, feature, cat_features, numerical_features, encoder, lb)
        assert(all([precision >= precision_threshold for precision in precisions])), f"Failed for {feature}" 
        assert(all([accuracy >= accuracy_threshold for accuracy in accuracies])),  f"Failed for {feature}"
        

def test_numerical_slice_precisions_accuracies():
    """ Test to see if a data slice of each feature of our dataset produces proper accuracies and precisions"""
    for feature in numerical_features:
        precisions, accuracies = validate_data_slice_performance(data_clean, model, feature, cat_features, numerical_features, encoder, lb)
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
