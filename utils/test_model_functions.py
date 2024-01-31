from .train_model import data, cat_features, numerical_features, model, encoder, lb, X_train
from .model import validate_data_slice_performance, compute_model_metrics, inference

precision_threshold = 0.6
accuracy_threshold = 0.7
tolerance = 0.0001

def test_categorical_slice_precisions_accuracies():
    """ Test to see if a data slice of each feature of our dataset produces proper accuracies and precisions"""
    for feature in cat_features:
        precisions, accuracies = validate_data_slice_performance(data, model, feature, cat_features, numerical_features, encoder, lb)
        assert(all([precision > precision_threshold for precision in precisions])), f"Failed for {feature}" 
        assert(all([accuracy > accuracy_threshold for accuracy in accuracies])),  f"Failed for {feature}"
        

def test_numerical_slice_precisions_accuracies():
    """ Test to see if a data slice of each feature of our dataset produces proper accuracies and precisions"""
    for feature in numerical_features:
        precisions, accuracies = validate_data_slice_performance(data, model, feature, cat_features, numerical_features, encoder, lb)
        assert(all([precision > precision_threshold for precision in precisions])), f"Failed for {feature}" 
        assert(all([accuracy > accuracy_threshold for accuracy in accuracies])),  f"Failed for {feature}"

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
    assert abs(fbeta-1) < tolerance #fbeta = 1, due to setting zero-division case to have value = 1

def test_inference_output_shape():
    """ Ensure that inference produces the same output shape as the input shape."""
    assert( len(inference(model, X_train)) == len(X_train) )
