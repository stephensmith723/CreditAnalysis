import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import * 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

## name list of classifiers
names = [
         "AdaBoost",
         "Bagging",
         "ExtraTrees",
         "GradientBoosting",
         "RandomForest",
         "Ridge",
         "SGD",
         "KNeighbors",
         "MLP",
         "DecisionTree",
         "LogisticRegression"
         ]

classifiers = [
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(n_estimators=158),
        GradientBoostingClassifier(learning_rate=0.15, n_estimators=158, max_depth=5),
        RandomForestClassifier(n_estimators=158),
        RidgeClassifier(),
        SGDClassifier(),
        KNeighborsClassifier(n_neighbors=5),
        MLPClassifier(hidden_layer_sizes = (200, 10), max_iter = 250, solver = "lbfgs"),
        DecisionTreeClassifier(),
        LogisticRegression(class_weight='balanced', solver='newton-cg', n_jobs=4)
        ]

classifier_no = len(classifiers)

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

def compute_measure(predicted_label, true_label):
    t_idx = (predicted_label == true_label) # truely predicted
    f_idx = np.logical_not(t_idx) # falsely predicted
    p_idx = (true_label > 0) # positive targets
    n_idx = np.logical_not(p_idx) # negative targets
    tp = np.sum( np.logical_and(t_idx, p_idx)) # TP
    tn = np.sum( np.logical_and(t_idx, n_idx)) # TN
    # false positive: original negative but classified as positive
    # false negative: original positive but classified as negative
    fp = np.sum(n_idx) - tn
    fn = np.sum(p_idx) - tp

    tp_fp_tn_fn_list=[]
    tp_fp_tn_fn_list.append(tp)
    tp_fp_tn_fn_list.append(fp)
    tp_fp_tn_fn_list.append(tn)
    tp_fp_tn_fn_list.append(fn)
    tp_fp_tn_fn_list=np.array(tp_fp_tn_fn_list)
    tp=tp_fp_tn_fn_list[0]
    fp=tp_fp_tn_fn_list[1]
    tn=tp_fp_tn_fn_list[2]
    fn=tp_fp_tn_fn_list[3]
    
    with np.errstate(divide='ignore'):
        sen = 0 if (tp+fn) == 0 else (1.0*tp)/(tp+fn)
    
    with np.errstate(divide='ignore'):
        spc = 0 if (tp+fp) == 0 else (1.0*tn)/(tn+fp)
    
    with np.errstate(divide='ignore'):
        ppr = 0 if (tp+fp) == 0 else (1.0*tp)/(tp+fp)
    
    with np.errstate(divide='ignore'):
        npr = 0 if (tp+fn) == 0 else (1.0*tn)/(tn+fn)
    
    with np.errstate(divide='ignore'):
        f1_score = 0 if ((2*tp)+fp+fn) == 0 else (2.0*tp)/((2*tp)+fp+fn)

    with np.errstate(divide='ignore'):
        acc = 0 if (tp+fp+tn+fn) == 0 else (tp+tn)*1.0/(tp+fp+tn+fn)

    with np.errstate(divide='ignore'):
        d_index = math.log(1 + acc, 2) + math.log(1 + ((sen + ppr)/2), 2)

    return d_index

def scaled(data):
    #All data
    data_scaled = preprocessing.scale(data)
    data_scaled = pd.DataFrame(data = data_scaled, columns = data.columns.values )
    return data_scaled

def datasplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)
    return X_train, X_test, y_train, y_test

def zero_mean_std(data_scaled):
    #All data
    data_zeromean = data_scaled.mean(axis=0)
    data_std = data_scaled.std(axis = 0)
    return data_zeromean, data_std

def run_models(data):
    summary = pd.DataFrame(data=[], columns=["Accuracy", "D-Index", "AUC-Score", "MSE"])
    sm = SMOTE(ratio = 'auto' , kind = 'regular')
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 3:-1], \
        data.iloc[:, -1], test_size = 0.2, random_state=42)
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    for name, clf in zip(names, classifiers):
        clf.fit(X_resampled, y_resampled)
        y_pred_train = clf.predict(X_resampled)
        y_pred = clf.predict(X_test)
        mse = mean_squared_error(y_test,y_pred)
        accuracy = clf.score(X_test, y_test)  #classification accuracy
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        auc_score = auc(fpr, tpr)
        train_cnf_matrix = confusion_matrix(y_resampled, y_pred_train)
        print(train_cnf_matrix)
        test_cnf_matrix = confusion_matrix(y_test, y_pred)
        print(test_cnf_matrix)
        com_measures = compute_measure(y_pred, y_test)
        measures = [accuracy, com_measures, auc_score, mse]
        summary = summary.append(pd.Series(measures, index=["Accuracy", "D-Index", "AUC-Score", "MSE"]), ignore_index=True)
    return summary

def main():
    data = pd.read_csv("dataset_diabetes/diabetic_data_cleaned.csv")
    # print(data.iloc[:, 3:-1].shape)
    # print(data.iloc[:, 3:-1].head())
    summary_model = run_models(data)
    summary_model.insert(loc=0, column='Model', value=names)
    print(summary_model)

if __name__ == '__main__':
    main()