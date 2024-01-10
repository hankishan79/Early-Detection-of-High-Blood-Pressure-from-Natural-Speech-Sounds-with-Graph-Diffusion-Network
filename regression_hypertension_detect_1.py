from datetime import time

import pandas as pd
from imblearn.combine import SMOTEENN  # Import SMOTE-ENN
from imblearn.over_sampling import SMOTE, ADASYN
from keras.optimizers import RMSprop
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.svm import LinearSVC
# import SVC classifier
from sklearn.svm import SVC
from sklearn import svm
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor, Ridge
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score # import KFold
import numpy as np
#import sklearn_relief as sr
from sklearn.metrics import roc_auc_score, roc_curve
# Import the library
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import TimeseriesGenerator

# Load data
data = pd.read_csv(r'C:\Users\hankishan\Desktop\S_H_B_Database_10132022\pyhton codes\new_train.csv')
data.head()

# Separate features and target
yi = data['SBP-2CLASSES']
X = data.drop(columns=['MIXED-CLASSES','DBP-2CLASSES', 'SBP-2CLASSES', 'SBP', 'DBP'], axis=1)
y = yi.map({1: 0, 2: 1})
# Apply SMOTE to training data only
#sm = SMOTE()
#X, y = sm.fit_resample(X, y)
# Apply ADASYN to training data only
#adasyn = ADASYN(sampling_strategy={1: 2500, 2: 2500}, n_neighbors=5, random_state=42)
#X, y = adasyn.fit_resample(X1, y1)
# Apply SMOTE-ENN to training data only
#smote_enn = SMOTEENN(sampling_strategy='auto', k_neighbors=5)
#X, y = smote_enn.fit_resample(X1, y1)
# Create a GroupShuffleSplit object
# Check the class distribution

group_split = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
# Define the features and target variables
features1 = X.drop('PAT_ID', axis=1)  # Assuming 'PAT_ID' is the column containing group information
target = y
# Generate the train/validation/test indices
train_idx, val_test_idx = next(group_split.split(features1, target, groups=X['PAT_ID']))
val_idx, test_idx = next(group_split.split(features1.iloc[val_test_idx], target.iloc[val_test_idx],
                                           groups=X.iloc[val_test_idx]['PAT_ID']))

# Split the data into train, validation, and test sets
X_train1, y_train1 = features1.iloc[train_idx], target.iloc[train_idx]
#adasyn = ADASYN(sampling_strategy={1: 2500, 2: 2500}, n_neighbors=5, random_state=42)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train1, y_train1)

X_val, y_val = features1.iloc[val_idx], target.iloc[val_idx]
X_test, y_test = features1.iloc[test_idx], target.iloc[test_idx]

# Print the shapes
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)

from featurewiz import FeatureWiz
features = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None,
                      verbose=2)
X_train = features.fit_transform(X_train, y_train)
X_test = features.transform(X_test)
X_val = features.transform(X_val)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
# With SMOTE Results ************************************************************************************
# Hyperparameter tuning for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['auto', 'scale', 0.1, 0.01],
    'kernel': ['linear', 'rbf', 'poly']
}
# Set probability=True for SVC
grid_search_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, n_jobs=7)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
print("SVM is OK\n")
# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=7)
grid_search_rf.fit(X_train, y_train)
best_rnd = grid_search_rf.best_estimator_
print("RND is OK\n")
# Hyperparameter tuning for Gradient Boosting Regressor
param_grid_gb = {
    'n_estimators': [1000],  # List of values to be tested
    'max_depth': [4],
    'min_samples_split': [2],
    'learning_rate': [0.01],  # Wrap the value in a list
    'loss': ['squared_error']
}
grid_search_gb = GridSearchCV(GradientBoostingRegressor(), param_grid_gb, cv=5, n_jobs=7)
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_
print("Gradient Boosting Regressor is OK\n")
# Hyperparameter tuning for Stochastic Gradient Descent Regressor
param_grid_sgdr = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
}
grid_search_sgdr = GridSearchCV(SGDRegressor(), param_grid_sgdr, cv=5, n_jobs=7)
grid_search_sgdr.fit(X_train, y_train)
best_sgdr = grid_search_sgdr.best_estimator_
print("Stochastic Gradient Descent Regressor is OK\n")
# Hyperparameter tuning for Ridge Regressor
param_grid_ridge = {
    'alpha': [0.1, 1, 10],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
}
grid_search_ridge = GridSearchCV(Ridge(), param_grid_ridge, cv=5, n_jobs=7)
grid_search_ridge.fit(X_train, y_train)
best_ridge = grid_search_ridge.best_estimator_
print("Ridge Regressor is OK\n")
# Ensemble regression approach late fusion
start = time()
# Make predictions for Stochastic Gradient Descent Regressor and Ridge Regressor
y_pred_rnd_val = best_rnd.predict(X_val)
y_pred_svm_val = best_svm.predict(X_val)
y_pred_gb_val = best_gb.predict(X_val)
y_pred_sgdr_val = best_sgdr.predict(X_val)
y_pred_ridge_val = best_ridge.predict(X_val)

y_pred_rnd_test = best_rnd.predict(X_test)
y_pred_svm_test = best_svm.predict(X_test)
y_pred_gb_test = best_gb.predict(X_test)
y_pred_sgdr_test = best_sgdr.predict(X_test)
y_pred_ridge_test = best_ridge.predict(X_test)
# Convert continuous predictions to class labels using a threshold
threshold_gb = 0.5  # Adjust this threshold as needed
y_pred_gb_val_classified = (y_pred_gb_val > threshold_gb).astype(int)
y_pred_gb_test_classified = (y_pred_gb_test > threshold_gb).astype(int)
# Convert continuous predictions to class labels using a threshold
threshold_sgdr = 0.5  # Adjust this threshold as needed
y_pred_sgdr_val_classified = (y_pred_sgdr_val > threshold_sgdr).astype(int)
y_pred_sgdr_test_classified = (y_pred_sgdr_test > threshold_sgdr).astype(int)
# Convert continuous predictions to class labels using a threshold
threshold_ridge = 0.5  # Adjust this threshold as needed
y_pred_ridge_val_classified = (y_pred_ridge_val > threshold_ridge).astype(int)
y_pred_ridge_test_classified = (y_pred_ridge_test > threshold_ridge).astype(int)
# Combine the predictions at the feature level
combined_features = np.column_stack((y_pred_ridge_val, y_pred_sgdr_val, y_pred_gb_val, y_pred_rnd_val, y_pred_svm_val))
combined_features_test = np.column_stack((y_pred_ridge_test, y_pred_sgdr_test, y_pred_gb_test, y_pred_rnd_test, y_pred_svm_test))
param_grid_gbF = {
    'n_estimators': 1000,  # List of values to be tested
    'max_depth': 4,
    'min_samples_split': 2,
    'learning_rate': 0.01,  # Wrap the value in a list
    'loss': 'squared_error'
}
# Train a late fusion model (Gradient Boosting Regressor in this case)
fusion_model = GradientBoostingRegressor(**param_grid_gbF)
fusion_model.fit(combined_features, y_val)
# Make predictions using the fusion model
ems_predict = fusion_model.predict(combined_features_test)
ems_score = fusion_model.score(combined_features_test, y_test)
threshold = 0.5  # Adjust this threshold as needed
ems_classified = (ems_predict > threshold).astype(int)
end = time()
print('SVM Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_svm_test)))
print('RND Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_rnd_test)))
print('GBR Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_gb_test_classified)))
print('SGDR Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_sgdr_test_classified)))
print('ridge Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred_ridge_test_classified)))
print('GBR-F Test accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, ems_classified)))

print('SVM Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_svm_test, zero_division=1)))
print('RND Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_rnd_test, zero_division=1)))
print('GBR Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_gb_test_classified, zero_division=1)))
print('SGDR Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_sgdr_test_classified, zero_division=1)))
print('ridge Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, y_pred_ridge_test_classified, zero_division=1)))
print('GBR-F Test accuracy score with default hyperparameters: {0:0.4f}', (classification_report(y_test, ems_classified, zero_division=1)))

from sklearn.preprocessing import label_binarize
#y_test = label_binarize(y_test, classes=[1, 2])[:, 0]

# Calculate the ROC curves for all models
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_test)
fpr_rnd, tpr_rnd, _ = roc_curve(y_test, y_pred_rnd_test)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb_test_classified)
fpr_sgdr, tpr_sgdr, _ = roc_curve(y_test, y_pred_sgdr_test_classified)
fpr_ridge, tpr_ridge, _ = roc_curve(y_test, y_pred_ridge_test_classified)
fpr_gbrF, tpr_gbrF, _ = roc_curve(y_test, ems_classified)

# Calculate AUC scores for all models
auc_svm = roc_auc_score(y_test, y_pred_svm_test)
auc_rnd = roc_auc_score(y_test, y_pred_rnd_test)
auc_gb = roc_auc_score(y_test, y_pred_gb_test_classified)
auc_sgdr = roc_auc_score(y_test, y_pred_sgdr_test_classified)
auc_ridge = roc_auc_score(y_test, y_pred_ridge_test_classified)
auc_gbrF = roc_auc_score(y_test, ems_classified)

# Plot the ROC curves for all models
plt.figure(figsize=(10, 10))
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})')
plt.plot(fpr_rnd, tpr_rnd, label=f'Random Forest (AUC = {auc_rnd:.2f})')
plt.plot(fpr_gb, tpr_gb, label=f'GBR (AUC = {auc_gb:.2f})')
plt.plot(fpr_sgdr, tpr_sgdr, label=f'SGDR (AUC = {auc_sgdr:.2f})')
plt.plot(fpr_ridge, tpr_ridge, label=f'Ridge (AUC = {auc_ridge:.2f})')
plt.plot(fpr_gbrF, tpr_gbrF, label=f'GBR-F (AUC = {auc_gbrF:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver Operating Characteristic (ROC) - All Models', fontsize=18)
plt.legend()
plt.show()


