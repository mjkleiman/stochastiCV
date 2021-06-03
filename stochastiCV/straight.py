import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (multilabel_confusion_matrix, confusion_matrix,  
    accuracy_score, recall_score, precision_score, f1_score)
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from ._utils import _multiclass_threshold, _model_repeat


class StochasticCV:
    '''
    model :  any scikit-learn classifier 
        Currently tested with randomforestclassifier, gradientboostedclassifier.
        Output metrics are tuned towards classifiers but future updates will 
        implement the ability to call regression models
    model_repeats :  int or list of ints
        Specify a list of seed values to be used, or specify an int to 
        deterministically generate that number of seeds
    num_classes : int
        Number of classes. If classes are not arranged in numerical format 
        (ex: 0,1,2) then specify class_labels
    class_labels : list of strings or ints
        Set labels of classes if not numerical from 0. Specifying class_labels 
        will disable num_classes
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
    verbose : 0, 1, or 2
        0 : disables all output
        1 : shows split/repeat number
        2 : adds confusion_matrix
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
        avg_strategy='macro', 
        verbose=0
    ):
        self.model = model
        self.num_classes = num_classes
        self.imbalanced_train = imbalanced_train
        self.imbalanced_test = imbalanced_test
        self.categorical_features = categorical_features
        self.over_strategy = over_strategy
        self.under_strategy = under_strategy
        self.avg_strategy = avg_strategy
        self.verbose = verbose
        
        if class_labels is None:
            self.class_labels = list(range(num_classes))
        else:
            self.class_labels = class_labels  

        if isinstance(model_repeats, int):
            self.model_repeats = list(int(x)*42+42 for x in range(model_repeats))
        else:
            self.model_repeats = model_repeats


    def fit_predict(self, X, y, X_test, y_test, threshold=None):
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
        '''
        df = pd.DataFrame()

        X_ = X
        y_ = y
        X_test_ = X_test
        y_test_ = y_test
        j = self.model_repeats[0]

        if self.imbalanced_train == 'over':
            if self.categorical_features is None:
                sm = SMOTE(random_state=j, sampling_strategy=self.over_strategy)
            else:
                sm = SMOTENC(categorical_features=self.categorical_features, sampling_strategy=self.over_strategy, random_state=j)
            X_, y_ = sm.fit_resample(X_,y_)
        if self.imbalanced_test == 'over':
            if self.categorical_features is None:
                sm = SMOTE(random_state=j, sampling_strategy=self.over_strategy)
            else:
                sm = SMOTENC(categorical_features=self.categorical_features, sampling_strategy=self.over_strategy, random_state=j)
            X_test_,y_test_ = sm.fit_resample(X_test_,y_test_)
        if self.imbalanced_train == 'under':
            enn = EditedNearestNeighbours(sampling_strategy=self.under_strategy)
            X_,y_ = enn.fit_resample(X_,y_)
        if self.imbalanced_test == 'under':
            enn = EditedNearestNeighbours(sampling_strategy=self.under_strategy)
            X_test_,y_test_ = enn.fit_resample(X_test_,y_test_)
        if self.imbalanced_train == 'overunder':
            if self.categorical_features is None:
                sm = SMOTE(random_state=j, sampling_strategy=self.over_strategy)
            else:
                sm = SMOTENC(categorical_features=self.categorical_features, sampling_strategy=self.over_strategy, random_state=j)
            enn = EditedNearestNeighbours(sampling_strategy=self.under_strategy)
            smenn = SMOTEENN(sampling_strategy=self.over_strategy, random_state=j, smote=sm, enn=enn)
            X_, y_ = smenn.fit_resample(X_,y_)
        if self.imbalanced_test == 'overunder':
            if self.categorical_features is None:
                sm = SMOTE(random_state=j, sampling_strategy=self.over_strategy)
            else:
                sm = SMOTENC(categorical_features=self.categorical_features, sampling_strategy=self.over_strategy, random_state=j)
            enn = EditedNearestNeighbours(sampling_strategy=self.under_strategy)
            smenn = SMOTEENN(sampling_strategy=self.over_strategy, random_state=j, smote=sm, enn=enn)
            X_test_,y_test_ = smenn.fit_resample(X_test_,y_test_)

        report = _model_repeat(X_, y_, X_test_, y_test_, threshold, self.model, self.model_repeats, self.num_classes, self.avg_strategy, j, self.verbose, self.class_labels)
        df = df.append(report)

        return df