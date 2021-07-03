StochastiCV
==============================

A method of cross-validation based on scikit-learn that splits data one or more times (using random or assigned seed values) and then repeats the model multiple times using different seeds. Current implementations include subsampling (simple train/test splits based on a ratio) and *k*-folds (split data into *k* splits, combining all but one for the train split and the left over one for the test split, repeated *k* times). Only classifier models are currently supported.

This function outputs metrics in a pandas DataFrame, including sensitivity (aka *recall*), specificity, positive predictive value (PPV) (aka *precision*), negative predictive value (NPV), f1_score, and overall accuracy, for each repeat of the model. Different model variations (e.g., hyperparmeter modification, feature sets, model architectures) can be statistically differentiated using ANOVAs or t-tests, facilitated by the reduced variance caused by large numbers of stochastic model repeats as compared to repeated shuffling of data.

If model stochasticity is desired (running model multiple times with varying outputs), only ensemble models and neural networks (e.g., multilayer perceptron) are supported. This is due to the use of stochasticity (randomness) within these types of models, to either assign initial weights (neural networks), in bootstrapping, or when searching for the best split at each node (ensemble models).

## API

A stochastiCV machine based on either subsampling or *k*-folds can be called, with required parameters including the scikit-learn model, number of times splits are repeated, number of times models are repeated, and the number of classes. In a *k*-folds  machine, the number of folds are also specified.
```
from stochastiCV import StochasticSubsamplingCV, StochasticKFoldsCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

# Subsampling
scv = StochasticSubsamplingCV(rf, split_repeats=5, model_repeats=5, num_classes=3, test_ratio=0.25)

# K Folds
scv = StochasticKFoldsCV(rf, folds=3, split_repeats=5, model_repeats=5, num_classes=3)
```
The machine is then ran by calling the ```.fit_predict()``` function. 
```
out = scv.fit_repeat(X, y)
```
The function will split the inputted data (X for features, y for
 classes) into *j* number of splits, multiplied by *k* folds if applicable. The model is also repeated *h* times, as specified. The data is stratified by default, but this can be disabled by setting ```stratify=False```. 

If you choose for data shuffling to be disabled, and you wish to specify a train and test set and only enable repeats due to model stochasticity, you may specify a test set as follows:
```
out = scv.fit_repeat(X_train, y_train, X_test=X_test, y_test=y_test)
```
This will disable the ```split_repeats``` parameter, and only run the model the amount of times specified in the ```model_repeats``` and (in *k*-folds) the ```folds``` parameters.

### Output
At present, the output is a pandas dataframe containing sensitivities, specificities, PPVs, NPVs, f1_scores, and 
accuracy metrics for both each individual class and averaged for all classes, with one row per repeat of the model. Confusion matrices for each repeat can be called by setting ```verbose=2``` (future updates will allow for these to be optionally output for additional processing).

### Averaging
Metrics are calculated for each individual class as well as the average of all classes. Averaging for metrics is identical to that used in scikit-learn's recall and precision functions: 'macro' finds the unweighted mean, 'micro' calculates global means, and 'weighted' calculates means for each class weighted by the number of instances of each class compared to the number of instances of all classes.

### Over- and under-sampling of splits
The option is provided for the train and/or test splits to be over- or under-sampled, using Synthetic Majority Oversampling TEchnique (SMOTE) for oversampling and Edited Nearest Neighbors (ENN) for undersampling, as implemented by *imbalanced-learn*. This is implemented directly into the machine to mitigate leakage that would be present if the the data was split after oversampling was performed.

Either ```imbalanced_train``` or ```imbalanced_test``` can be set to 'over' (oversampling), 'under' (for undersampling), or 'overunder' to first oversample and then undersample, a technique that leads to more even class proportions and less overfitting by first oversampling the minority class by a small amount (e.g. 10%) and then undersampling the majority class by a larger amount (e.g. 30%), resulting in less artificially created datapoints yet more even class sizes. If categorical features are used, they should be specified by setting ```categorical_features``` to a list of feature labels. *Imbalanced-learn*'s categorical SMOTE function (SMOTENC) will then be used.

The strategies for SMOTE and ENN are used to specify the degree of over- or under-sampling performed. See ```imblearn.over_sampling.SMOTE``` documentation for ```sampling_strategy```, as this is passed to our parameter ```over_strategy```. For the parameter ```under_strategy```, see ```imblearn.under_sampling.EditedNearestNeighbours```.  

### Initial splits
Occasionally, when data is limited, optimization or other processing must be performed on a subset of the data. If this data is added to the testing set, leakage can occur that would impact the ability to utilize output metrics (e.g., accuracy) in comparison to models where leakage did not occur. The option for an "initial split" was therefore implemented that enables the ```sklearn.model_selection.train_test_split``` function to be performed prior to any other data splitting, with this split added back in to the training set only to ensure that no data is wasted.

For example, if you have 100 points of data, but you wish to use 25% to initially identify the model's optimal hyperparameters, you could perform a ```sklearn.model_selection.train_test_split``` function with the parameters ```train_ratio=0.25``` and ```random_seed=42``` outside of the stochastiCV machine. Once ideal hyperparamters are identified and the model is initialized with them, the stochastiCV model ```StochasticSubsamplingCV``` or ```StochasticKFoldsCV``` could be called with the parameters ```initial_split_ratio=0.25``` and ```initial_split_seed=42```. Thus, the same data points would be initially removed (before splitting occurs) and then added back into every train set, ensuring that data used in hyperparameter optimization is not wasted but also never appears in the test set.

### Thresholding
In some situations, it is beneficial to manually set the threshold for determining a class based on its predicted probabilities. We have enabled the ability to do so by assigning the ```threshold``` parameter within the ```.fit_predict()``` function for either each individual class  (pass a list of floats) or overall (pass a single float, binary only). This enables weaker classifier to be given more weight, allowing accuracy metrics to be better dialed in to a more favorable result. 

For example, in some medical applications, specificity (negative class recall) is more useful than sensitivity (positive class recall) -- assigning the threshold to be a lower number may result in higher specificities at the expense of lower sensitivities.

## To Do:
- Add compatibility with regression models including RandomForestRegressor, and include outputs applicable to regressors
- Enable output of overall confusion matrix: option for either sum total of all repeats, or averaged 
- Implementation of Leave One Out CV (model repeats only)
- Addition of area under ROC and Precision-Recall curves
- Keras and PyTorch implementations