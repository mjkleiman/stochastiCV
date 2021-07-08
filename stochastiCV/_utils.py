import pandas as pd
import numpy as np
from sklearn.metrics import (multilabel_confusion_matrix, confusion_matrix,  
    accuracy_score, recall_score, precision_score, f1_score)

def _multiclass_threshold(y_pred, threshold, class_labels):
    condition_list = []
    for i in class_labels:
        # Create condition for each in threshold, append condition to condition_list
        condition_list.append((y_pred[:,i] >= threshold[i]))
    y_pred = np.select(condition_list.reverse(), class_labels.reverse())

    return y_pred


def _model_repeat_clf(X_, y_, X_test_, y_test_, threshold, model, model_repeats, num_classes, avg_strategy, j, verbose, class_labels):
    df = pd.DataFrame()

    for h in model_repeats:
        np.random.seed(h)
        model.set_params(random_state=h)
        model.fit(X_,y_)
        y_pred = model.predict_proba(X_test_)

        # Modify classification thresholds
        if isinstance(threshold, float):
            if 0 <= threshold <= 1:
                y_pred = np.where(y_pred[:,1] >= threshold, 1, 0)
            else:
                raise ValueError('Thresholds must be set to a number between 0 and 1')
        elif isinstance(threshold, list):
            if 0 <= threshold <= 1:
                y_pred = _multiclass_threshold(y_pred, threshold, class_labels)
            else:
                raise ValueError('Thresholds must be set to a number between 0 and 1')
        elif threshold is None:
            y_pred = np.argmax(y_pred, axis=1) # Default, selection via highest score
        cmat = multilabel_confusion_matrix(np.array(y_test_), y_pred) # Generate confusion matrix per run

        if verbose >= 1:
            print('Split: ',j,', Model: ',h)
        if verbose >= 2:
            if num_classes == 2:
                print(cmat[1])
            if num_classes >= 3:
                print(cmat)
        
        report = pd.DataFrame({'j':j,'h':h}, index=[0]) 
        
        # Report statistics for each class, write to report dataframe
        for c in range(num_classes):                
            TP = cmat[c][1][1]
            FP = cmat[c][0][1]
            TN = cmat[c][0][0]
            FN = cmat[c][1][0]
                            
            report['Sensitivity'+str(c)] = (TP/(TP+FN))
            report['Specificity'+str(c)] = (TN/(FP+TN))
            report['PPV'+str(c)] = (TP/(TP+FP))
            report['NPV'+str(c)] = (TN/(TN+FN))
            report['Accuracy'+str(c)] = (TP+TN)/len(y_test_)

        # Report overall statistics, write to report dataframe
        if num_classes == 2:
            report['Sensitivity'] = recall_score(y_test_,y_pred, average=avg_strategy, labels=[1])
            report['Specificity'] = recall_score(y_test_,y_pred, average=avg_strategy, labels=[0])
            report['PPV'] = precision_score(y_test_,y_pred, average=avg_strategy, labels=[1])
            report['NPV'] = precision_score(y_test_,y_pred, average=avg_strategy, labels=[0])
            report['F1_Score'] = f1_score(y_test_,y_pred, average=avg_strategy)
            report['Accuracy'] = accuracy_score(y_test_, y_pred)
        elif num_classes >= 3:
                if avg_strategy == 'macro':
                    report['Sensitivity'] = report.filter(regex='Sensitivity').mean(axis=1)
                    report['Specificity'] = report.filter(regex='Specificity').mean(axis=1)
                    report['PPV'] = report.filter(regex='PPV').mean(axis=1)
                    report['NPV'] = report.filter(regex='NPV').mean(axis=1)
                    report['F1_Score'] = f1_score(y_test_,y_pred, average=avg_strategy)
                    report['Accuracy'] = report.filter(regex='Accuracy').mean(axis=1)
                if avg_strategy == 'weighted':
                    # Report weighted Sensitivity using sklearn's recall_score
                    report['Sensitivity'] = recall_score(y_test_,y_pred,average=avg_strategy)
                    # Calculate weighted Specificity manually by assigning proportionally higher weights to classes with fewer subjects 
                    r_specificity = []
                    for c in range(num_classes):
                        num = class_labels.copy()
                        num.pop(c)
                        r_specificity.append(recall_score(y_test_,y_pred,average=avg_strategy, labels=num))
                    report['Specificity'] = pd.Series(r_specificity).mean()
                    # Report weighted PPV using sklearn's precision_score
                    report['PPV'] = precision_score(y_test_,y_pred,average=avg_strategy)
                    # Calculate weighted NPV manually by assigning proportionally higher weights to classes with fewer subjects
                    r_npv = []
                    for c in range(num_classes):
                        num = class_labels.copy()
                        num.pop(c)
                        r_npv.append(precision_score(y_test_,y_pred,average=avg_strategy, labels=num))
                    report['NPV'] = pd.Series(r_npv).mean()
                    report['F1_Score'] = f1_score(y_test_,y_pred, average=avg_strategy)
                    report['Accuracy'] = accuracy_score(y_test_, y_pred)
        
        report.set_index(['j','h'], inplace=True)
        df = df.append(report)

    return df



