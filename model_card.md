# Model Card

## Model Details
Shawn Y created the model. It is an Adaboost ensemble classifier model using the default hyperparameters in scikit-learn 1.3.2.

## Intended Use
This model should be used to predict the income range of an individual based off a handful of attributes.

## Training Data
The data originated form the UCI ML repository (https://archive.ics.uci.edu/dataset/20/census+income) and represents data from a 1994 Census database.

## Evaluation Data
The model was evaluated/tested to ensure it's not overfitting with a test dataset that was 20% of the original dataset. 
Additionally, data slice subsets were evaluated for all categorical and all numerical features to ensure that their metrics fall into an acceptable range. For numerical features, 2 slice subsets were tested above and below the median for each numerical feature. For categorical features, all slice subsets with frequencies greater than a count of 1000 (>= ~3% of overall dataset) were assessed.

## Metrics
The overall metrics used included precision, recall, accuracy, and f1-score. 
The overall model's performance on the test set yielded the following scores:
Test Precision: 0.77, Test Recall: 0.61, Test F1-Score: 0.68, Test Accuracy: 0.86 
For assessing the performance of each slice, the accuracy of each categorical slice was tested against a threshold of 0.7. For numerical features, each slice was assessed against both precision and accuracy thresholds of 0.55 and 0.7 respectively.

## Ethical Considerations
By ensuring that the precision and accuracy for each slice exceeded values of 0.6 and 0.7 respectively, we can deem that the model is ethically sound and isn't excessively biased towards any one slice.

## Caveats and Recommendations
The model performance may be improved via the use of a deep-learning algorithm and/or hyperparameter tuning, as a relatively simple ensemble learning algorithm is used.