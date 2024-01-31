import pytest
from train_model import data, cat_features, numerical_features, encoder, lb
from ml.model import validate_data_slice_performance

precision_threshold = 0.6
accuracy_threshold = 0.75

def test_slice_precisions_accuracies():
    """ Test to see if a data slice of each feature of our dataset produces proper accuracies and precisions"""
    precisions, accuracies = validate_data_slice_performance(data, model, feature, categorical_features, numerical_features, encoder, lb)
    pass