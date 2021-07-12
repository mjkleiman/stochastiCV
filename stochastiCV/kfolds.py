import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (multilabel_confusion_matrix, confusion_matrix,  
    accuracy_score, recall_score, precision_score, f1_score)
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from ._utils import (_model_repeat_clf, 
    _model_repeat_reg, _model_ID, _clf_balancing)


class StochasticKFoldsCV:
    '''
    model :  any scikit-learn model 
        Currently tested with randomforestclassifier, gradientboostedclassifier,
        randomforestregressor, gradientboostedregressor.
        Model stochasticity only functions in models where random_state 
        modification alters outputs.
    model_type : string (default = 'auto')
        Specify whether the model is a classifier or regressor
        Set equal to 'clf' for classifier, or 'reg' for regressor.
        Default 'auto' will automatically detect some scikit-learn models, 
        but if this fails you may specify the type manually to bypass the check.
    folds : int
        Specify number of folds that dataset is split into
    split_repeats :  int or list of ints
        Determine number of times KFolds splits will be performed.
        Specify a list of seed values to be used, or specify an int to 
        deterministically generate that number of seeds.
        Splitting will always occur in this function. To disable splitting, use
        the straight.StochastiCV function
    model_repeats :  int or list of ints
        Specify a list of seed values to be used, or specify an int to 
        deterministically generate that number of seeds
    num_classes : int
        Number of classes. If classes are not arranged in numerical format 
        (ex: 0,1,2) then specify class_labels.
        Ignored if model_type set to 'reg'.
    class_labels : list of strings or ints
        Set labels of classes if not numerical from 0. Specifying class_labels 
        will disregard num_classes and derive number of classes from the labels.
        Ignored if model_type set to 'reg'.
    imbalanced_train : default=None
        'over' : utilize imbalanced-learn's SMOTE (or SMOTENC if 
            categorical_features are defined) to oversample the train set
        'under' : utilize imbalanced-learn's EditedNearestNeighbours to 
            undersample the train set
        'overunder' : oversample the train set using SMOTE (or SMOTENC if 
            categorical_features are defined) and then undersample using ENN
    imbalanced_test : default=None
        'over' : utilize imbalanced-learn's SMOTE (or SMOTENC if 
            categorical_features are defined) to oversample the test set 
            (not recommended)
        'under' : utilize imbalanced-learn's EditedNearestNeighbours to 
            undersample the test set
        'overunder' : oversample the test set using SMOTE (or SMOTENC if 
            categorical_features are defined) and then undersample using ENN
    over_strategy : see "search_strategy" from imblearn.oversampling.SMOTE
    under_strategy : see "search_strategy" from 
        imblearn.undersampling.EditedNearestNeighbours
    categorical_features : list of categorical features in data, used in SMOTENC
    avg_strategy : see 'average' in sklearn's roc_auc_score (default = 'macro')
    initial_split_seed : int
        If this value is specified, data will be initially split once. Use this 
        to match previously used train/test splits (sklearn implementation) and 
        to ensure that training data remains in the training set. Data on the 
        testing side of the split may be shuffled into the training/testing 
        sets, but the train side of the initial split will never appear in this 
        function's test set. If this value is not specified, all data will be 
        shuffled. This initial split is useful if a holdout test set will be 
        used for final testing. This can also be used to ensure all train sets
        see the same subset of data that never appears in the test set.
    initial_split_ratio : float
        If initial_split_seed is specified, this ratio will be used to split 
        initial train/test ratios. Small train splits are preferred to enable 
        more data to be shuffled and to reduce overfitting.
        This value replaces "train_size" in sklearn's train_test_split.
        NOTE: the train data from this initial split will be added to all 
        training sets generated 
    '''
    def __init__(self,
        model,
        folds=3,
        split_repeats=3,
        model_repeats=3,
        num_classes=2,
        class_labels=None,
        imbalanced_train=None, 
        imbalanced_test=None, 
        over_strategy='auto', 
        under_strategy='auto',
        categorical_features=None,
        avg_strategy='macro', 
        initial_split_seed=None, 
        initial_split_ratio=0.25
    ):
        self.model = model
        self.model_type = model_type
        self.folds = folds
        self.num_classes = num_classes
        self.imbalanced_train = imbalanced_train
        self.imbalanced_test = imbalanced_test
        self.categorical_features = categorical_features
        self.over_strategy = over_strategy
        self.under_strategy = under_strategy
        self.avg_strategy = avg_strategy
        self.initial_split_seed = initial_split_seed
        self.initial_split_ratio = initial_split_ratio
        
        if class_labels is None:
            self.class_labels = list(range(num_classes))
        else:
            self.class_labels = class_labels  

        if isinstance(split_repeats, int):
            self.split_repeats = list(int(x)*42+42 for x in range(split_repeats))
        else:
            self.split_repeats = split_repeats

        if isinstance(model_repeats, int):
            self.model_repeats = list(int(x)*42+42 for x in range(model_repeats))
        else:
            self.model_repeats = model_repeats

        if model_type == 'auto':
            self.model_type = _model_ID(model)
        elif model_type in ('clf', 'reg'):
            self.model_type = model_type
        else:
            raise TypeError('Variable model_type must indicate either a classifier (clf) or regressor (reg)')


    def fit_predict(self, X, y, X_test=None, y_test=None, threshold=None, stratify=True, raw_output=False, verbose=0):
        '''
        X : pandas DataFrame
        y : pandas Series or numpy array
        X_test : pandas DataFrame
        y_test : pandas Series or numpy array
        threshold : float (binary only) or list of floats (multiclass)
            Threshold applied to predicted probabilities (predict_proba in 
            sklearn), where the given class is selected if its probability is 
            greater than or equal to the given threshold number.
            Ignored if model_type set to 'reg'.

            If a float is given, it will be used as the threshold for predicting
                the positive class. 
            If a list of floats is given, each will be used for the respective 
            class, in order.
                NOTE: The default threshold is 0.5, so pass this number if you 
                only wish to specify a threshold for one class but not others
                Example: If you have three classes, and wish to predict the 
                third class ("2") if it is given a score of .3 or greater, you 
                should specify: "threshold = [.5, .5, .3]"
        stratify : bool (default=True)
            If True, preserve proportions of classes within splits. Randomize 
            splits if False.
        raw_output : bool (default=False)
            Currently does nothing. Future versions will enable for raw outputs
            of predicted vs true labels, for manual processing (e.g. ROC plots)
        verbose : 0, 1, or 2
            0 : disables all output
            1 : shows split/repeat number
            2 : adds confusion_matrix
        '''
        df = pd.DataFrame()
        X = np.asarray(X) # Ensure formatted as numpy array
        y = np.asarray(y) # Ensure formatted as numpy array

        if self.initial_split_seed is not None:
            if stratify is True:
                _X_train, X, _y_train, y = train_test_split(X, y.values.ravel(), train_size=self.initial_split_ratio, random_state=self.initial_split_seed, stratify=y)
            elif stratify is False:
                _X_train, X, _y_train, y = train_test_split(X, y.values.ravel(), train_size=self.initial_split_ratio, random_state=self.initial_split_seed, stratify=None)
            y = pd.Series(y)

        if self.model_type == 'clf':
            for j in self.split_repeats:
                if stratify is True:
                    kfold = StratifiedKFold(n_splits=self.folds, random_state=j, shuffle=True)
                elif stratify is False:
                    kfold = KFold(n_splits=self.folds, random_state=j, shuffle=True)
                for train_index, test_index in kfold.split(X,y):
                    X_, X_test_ = X[train_index], X[test_index]
                    y_, y_test_ = y[train_index], y[test_index]
                    if self.initial_split_seed is not None:
                        X_ = X_.append(_X_train)
                        y_ = np.append(y_,_y_train)
                    if X_test is not None:
                        X_test_ = X_test_.append(X_test)
                        y_test_ = np.append(y_test_,y_test)                

                    X_, y_, X_test_, y_test_ = _clf_balancing(
                        X_, y_, X_test_, y_test_, j, self.imbalanced_train, 
                        self.imbalanced_test, self.over_strategy, 
                        self.under_strategy, self.categorical_features)

                    # Run models
                    report = _model_repeat_clf(X_, y_, X_test_, y_test_, threshold, self.model, self.model_repeats, self.num_classes, self.avg_strategy, j, verbose, self.class_labels)
                    df = df.append(report)

        if self.model_type == 'reg':
            for j in self.split_repeats:
                kfold = KFold(n_splits=self.folds, random_state=j, shuffle=True)
                for train_index, test_index in kfold.split(X,y):
                    X_, X_test_ = X[train_index], X[test_index]
                    y_, y_test_ = y[train_index], y[test_index]
                    if self.initial_split_seed is not None:
                        X_ = X_.append(_X_train)
                        y_ = np.append(y_,_y_train)
                    if X_test is not None:
                        X_test_ = X_test_.append(X_test)
                        y_test_ = np.append(y_test_,y_test)

                    # Placeholder location for future regression augmentation/balancing

                    report = _model_repeat_reg(X_, y_, X_test_, y_test_, self.model, self.model_repeats, j, verbose)
                    df = df.append(report)

        return df
