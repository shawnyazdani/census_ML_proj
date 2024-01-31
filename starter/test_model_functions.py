import pytest
from train_model import data, cat_features, numerical_features, model, encoder, lb
from ml.model import validate_data_slice_performance

precision_threshold = 0.6
accuracy_threshold = 0.7

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
        