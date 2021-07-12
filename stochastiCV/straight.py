import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (multilabel_confusion_matrix, confusion_matrix,  
    accuracy_score, recall_score, precision_score, f1_score)
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from ._utils import (_model_repeat_clf, 
    _model_repeat_reg, _model_ID, _clf_balancing)


class StochasticMachine:
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
    model_repeats :  int or list of ints
        Specify a list of seed values to be used, or specify an int to 
        deterministically generate that number of seeds
    num_classes : int
        Number of classes. If classes are not arranged in numerical format 
        (ex: 0,1,2) then specify class_labels
        Ignored if model_type set to 'reg'
    class_labels : list of strings or ints
        Set labels of classes if not numerical from 0. Specifying class_labels 
        will disable num_classes
        Ignored if model_type set to 'reg'
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
    '''
    def __init__(self,
        model,
        model_repeats=3,
        num_classes=2,
        class_labels=None,
        imbalanced_train=None, 
        imbalanced_test=None, 
        over_strategy='auto', 
        under_strategy='auto',
        categorical_features=None,
        avg_strategy='macro'
    ):
        self.model = model
        self.num_classes = num_classes
        self.imbalanced_train = imbalanced_train
        self.imbalanced_test = imbalanced_test
        self.categorical_features = categorical_features
        self.over_strategy = over_strategy
        self.under_strategy = under_strategy
        self.avg_strategy = avg_strategy
        
        if class_labels is None:
            self.class_labels = list(range(num_classes))
        else:
            self.class_labels = class_labels  

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


    def fit_predict(self, X, y, X_test, y_test, threshold=None, raw_output=False, verbose=0):
        '''
        X : pandas DataFrame
        y : pandas Series or numpy array
        X_test : pandas DataFrame
        y_test : pandas Series or numpy array
        threshold : float (binary only) or list of floats (multiclass)
            Threshold applied to predicted probabilities (predict_proba in 
            sklearn), where the given class is selected if its probability is 
            greater than or equal to the given threshold number.
            If a float is given, it will be used as the threshold for predicting
                the positive class. 
            If a list of floats is given, each will be used for the respective 
            class, in order.
                NOTE: The default threshold is 0.5, so pass this number if you 
                only wish to specify a threshold for one class but not others
                Example: If you have three classes, and wish to predict the 
                third class ("2") if it is given a score of .3 or greater, you 
                should specify: "threshold = [.5, .5, .3]"
        raw_output : bool (default=False)
            Currently does nothing. Future versions will enable for raw outputs
            of predicted vs true labels, for manual processing (e.g. ROC plots)
        verbose : 0, 1, or 2
            0 : disables all output
            1 : shows split/repeat number
            2 : adds confusion_matrix
        '''
        df = pd.DataFrame()

        X_ = X
        y_ = y
        X_test_ = X_test
        y_test_ = y_test
        j = self.model_repeats[0]

        if self.model_type == 'clf':
            X_, y_, X_test_, y_test_ = _clf_balancing(
                    X_, y_, X_test_, y_test_, j, self.imbalanced_train, 
                    self.imbalanced_test, self.over_strategy, 
                    self.under_strategy, self.categorical_features)

            report = _model_repeat_clf(X_, y_, X_test_, y_test_, threshold, self.model, self.model_repeats, self.num_classes, self.avg_strategy, j, verbose, self.class_labels)
            df = df.append(report)

        if self.model_type == 'reg':
            # Placeholder location for future regression augmentation/balancing

            report = _model_repeat_reg(X_, y_, X_test_, y_test_, self.model, self.model_repeats,j, verbose)
            df = df.append(report)

        return df
