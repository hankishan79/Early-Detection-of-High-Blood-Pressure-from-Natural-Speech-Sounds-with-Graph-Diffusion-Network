from time import time
import pandas as pd
from joblib import parallel_backend
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, \
    r2_score
import seaborn as sns
from sklearn.svm import LinearSVC, SVR
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
from sklearn.linear_model import LogisticRegression, SGDRegressor, Ridge, LinearRegression
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
#from mlens.ensemble import BlendEnsemble

from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor
from hpsklearn import any_preprocessing
from hyperopt import tpe
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


def myround(x, base, div, subt):
    return (base * round(x / base) / div) - subt

def dividex(data, x, rem):
    return (data/x)-rem
svm=[]
ens=[]
ss=[]
rnd=[]
rig=[]
gebr=[]
segdr=[]
gpr=[]
ens.append(0)
ss.append(0)
svm.append(0)
rnd.append(0)
rig.append(0)
gebr.append(0)
segdr.append(0)
ens_f=[]
ss_f=[]
svm_f=[]
rnd_f=[]
rig_f=[]
gebr_f=[]
segdr_f=[]

gpr_f=[]
ens_p=[]
ss_p=[]
svm_p=[]
rnd_p=[]
rig_p=[]
gebr_p=[]
segdr_p=[]
ens_p.append(0)
ss_p.append(0)
svm_p.append(0)
rnd_p.append(0)
rig_p.append(0)
gebr_p.append(0)
segdr_p.append(0)

ens_f.append(0)
ss_f.append(0)
svm_f.append(0)
rnd_f.append(0)
rig_f.append(0)
gebr_f.append(0)
segdr_f.append(0)

ens_feature=[]
ens_feature.append(0)
ss_feature=[]
ss_feature.append(0)
svm_feature=[]
svm_feature.append(0)
rnd_feature=[]
rnd_feature.append(0)
rig_feature=[]
rig_feature.append(0)
gebr_feature=[]
gebr_feature.append(0)
segdr_feature=[]
segdr_feature.append(0)

ens_mse=[]
ss_mse=[]
svm_mse=[]
rnd_mse=[]
rig_mse=[]
gebr_mse=[]
segdr_mse=[]
ens_mse.append(0)
ss_mse.append(0)
svm_mse.append(0)
rnd_mse.append(0)
rig_mse.append(0)
gebr_mse.append(0)
segdr_mse.append(0)

ens_mae=[]
ss_mae=[]
svm_mae=[]
rnd_mae=[]
rig_mae=[]
gebr_mae=[]
segdr_mae=[]
ens_mae.append(0)
ss_mae.append(0)
svm_mae.append(0)
rnd_mae.append(0)
rig_mae.append(0)
gebr_mae.append(0)
segdr_mae.append(0)

ens_spearman=[]
ss_spearman=[]
svm_spearman=[]
rnd_spearman=[]
rig_spearman=[]
gebr_spearman=[]
segdr_spearman=[]
ens_spearman.append(0)
ss_spearman.append(0)
svm_spearman.append(0)
rnd_spearman.append(0)
rig_spearman.append(0)
gebr_spearman.append(0)
segdr_spearman.append(0)

ens_pearson=[]
ss_pearson=[]
svm_pearson=[]
rnd_pearson=[]
rig_pearson=[]
gebr_pearson=[]
segdr_pearson=[]
ens_pearson.append(0)
ss_pearson.append(0)
svm_pearson.append(0)
rnd_pearson.append(0)
rig_pearson.append(0)
gebr_pearson.append(0)
segdr_pearson.append(0)

ens_kendal=[]
ss_kendal=[]
svm_kendal=[]
rnd_kendal=[]
rig_kendal=[]
gebr_kendal=[]
segdr_kendal=[]
ens_kendal.append(0)
ss_kendal.append(0)
svm_kendal.append(0)
rnd_kendal.append(0)
rig_kendal.append(0)
gebr_kendal.append(0)
segdr_kendal.append(0)

ens_expvar=[]
ss_expvar=[]
svm_expvar=[]
rnd_expvar=[]
rig_expvar=[]
gebr_expvar=[]
segdr_expvar=[]
ens_expvar.append(0)
ss_expvar.append(0)
svm_expvar.append(0)
rnd_expvar.append(0)
rig_expvar.append(0)
gebr_expvar.append(0)
segdr_expvar.append(0)


for k in range(1):
    v=k/10
    for l in range(1):
        data = pd.read_csv(r'C:\Users\hankishan\Desktop\S_H_B_Database_10132022\pyhton codes\new_train.csv')
        data.head()
        y = data['SBP-2CLASSES']
        X = data.drop(columns=['MIXED-CLASSES'])
        from imblearn.over_sampling import SMOTE

        sm = SMOTE()
        X, y = sm.fit_resample(X, y)

        y_train = X['SBP']
        #X=X[['WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER']]
        #X = X.drop(['PAT_ID', 'DBP-2CLASSES', 'SBP-2CLASSES', 'SBP', 'DBP','WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER'], axis=1)
        X = X.drop(['PAT_ID', 'DBP-2CLASSES', 'SBP-2CLASSES', 'SBP','BPM', 'DBP'], axis=1)
        data_2 = pd.read_csv(r'C:\Users\hankishan\Desktop\S_H_B_Database_10132022\pyhton codes\new_test.csv')
        X_other = data_2.drop(columns=['DBP-2CLASSES', 'MIXED-CLASSES', 'SBP-2CLASSES', 'SBP','BPM', 'DBP'])
        #X_other = data_2.drop(columns=['DBP-2CLASSES', 'MIXED-CLASSES', 'SBP-2CLASSES', 'SBP', 'DBP','WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER'])
        #X_other = data_2[['WEIGHT', 'AGE', 'BPM', 'HEIGHT', 'GENDER','PAT_ID']]
        y_other = data_2['SBP']

        # Shuffle and dividing data as train and test for first stage *******
        gs = GroupShuffleSplit(n_splits=2, test_size=.50, random_state=0)
        train_ix, test_ix = next(gs.split(X_other, y_other, groups=X_other['PAT_ID']))
        X_test = X_other.loc[train_ix]
        y_test = y_other.loc[train_ix]
        X_val = X_other.loc[test_ix]
        y_val = y_other.loc[test_ix]

        # X_test_other, y_test_other = shuffle(X_test_other, y_test_other)
        X_test = X_test.drop(['PAT_ID'], axis=1)
        X_val = X_val.drop(['PAT_ID'], axis=1)

        from featurewiz import FeatureWiz
        features = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None,
                              verbose=0)
        X_train = features.fit_transform(X, y)
        X_test = features.transform(X_test)
        X_val = features.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        y_train = myround(y_train, 10, 10, 4)
        y_test = myround(y_test, 10, 10, 4)
        y_val = myround(y_val, 10, 10, 4)

        print("train Min : ", y_train.min(), "Max : ", y_train.max())
        print("test Min : ", y_test.min(), "Max : ", y_test.max())
        print("val Min : ", y_val.min(), "Max : ", y_val.max())
        print("count : ", k, l)

        start = time()
        # SVR support vector regression*******************
        regr = RandomForestRegressor(n_estimators=100, max_depth=50, min_samples_split=2, oob_score = True, n_jobs=7)
        regr.fit(X_train, y_train)
        svr_trn=regr.predict(X_train)
        y_pred_rndr = regr.predict(X_val)
        y_pred_rndr_test = regr.predict(X_test)
        score_rndr = regr.score(X_test, y_test)
        end = time()
        # report execution time
        result = end - start
        print('%.3f seconds' % result)

        start = time()
        # Linear Regression ********************************
        regr_lr = SVR(kernel='rbf')
        regr_lr.fit(X_train, y_train)
        y_pred_lr = regr_lr.predict(X_val)
        y_pred_lr_test = regr_lr.predict(X_test)
        score_lr = regr_lr.score(X_test, y_test)
        end = time()
        result = end - start
        print('%.3f seconds' % result)
        start = time()
        # Hyperparameters for GradientBoostingRegressor
        gbr_params = {'n_estimators': 1000,
                      'max_depth': 4,
                      'min_samples_split': 2,
                      'learning_rate': 0.01,
                      'loss': 'squared_error'}
        gbr = GradientBoostingRegressor(**gbr_params)
        with parallel_backend('threading'):  # 'threading' for multi-threading, 'multiprocessing' for multi-processing
            gbr.fit(X_train, y_train)
        #gbr.fit(X_train, y_train)
        y_pred_gbr = gbr.predict(X_val)
        y_pred_gbr_test = gbr.predict(X_test)
        score_gbr = gbr.score(X_test, y_test)
        end = time()
        result = end - start
        print('%.3f seconds' % result)
        start = time()
        # stochastic gradient descent regressor
        regr_sgdr = SGDRegressor(alpha=0.1, epsilon=0.1, eta0=0.1, penalty='elasticnet')
        regr_sgdr.fit(X_train, y_train)
        y_pred_sgdr = regr_sgdr.predict(X_val)
        y_pred_sgdr_test = regr_sgdr.predict(X_test)
        score_sgdr = regr_sgdr.score(X_test, y_test)
        end = time()
        #RİDGE MODEL *****************************************************************************************
        rigr = Ridge()
        # Define the parameter grid for the grid search
        param_grid = {
            'ridge__alpha': [1.0, 10.0],
            'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
        # Create a pipeline with standard scaling and Ridge regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', rigr)
        ])
        # Create a GridSearchCV object with parallel processing
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=7)
        # Fit the model to the data
        grid_search.fit(X_train, y_train)
        # Access the best estimator and best parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        y_pred_rigr = best_model.predict(X_val)
        # Make predictions
        y_pred_rigr_test = best_model.predict(X_test)
        score_rigr = best_model.score(X_test, y_test)
        end = time()
        result = end - start
        print('%.3f seconds' % result)

        #*********************** Self Supervise Model ******************************************************************
        input_dim = X_train.shape[1]
        latent_dim = 64
        # Define the autoencoder model architecture
        autoencoder = Sequential()
        autoencoder.add(Dense(units=128, activation='relu', input_shape=(input_dim,)))
        autoencoder.add(Dense(units=64, activation='relu'))
        autoencoder.add(Dense(units=latent_dim, activation='relu'))
        autoencoder.add(Dense(units=64, activation='relu'))
        autoencoder.add(Dense(units=128, activation='relu'))
        autoencoder.add(Dense(units=input_dim, activation='sigmoid'))
        # Compile the model
        autoencoder.compile(optimizer='SGD', loss='mean_squared_error')
        # Train the autoencoder
        autoencoder.fit(X_train, X_train, epochs=100, batch_size=32,verbose=0)
        # Create a new model using the encoder part of the trained autoencoder
        encoder = Sequential()
        encoder.add(autoencoder.layers[0])
        encoder.add(autoencoder.layers[1])
        encoder.add(autoencoder.layers[2])
        # Extract the learned representations (latent space) for the training and testing data
        encoded_train = encoder.predict(X_train)
        encoded_val = encoder.predict(X_val)
        encoded_test = encoder.predict(X_test)
        # Define a regression model (e.g., a simple feedforward neural network)
        regressor = Sequential()
        regressor.add(Dense(units=64, activation='relu', input_shape=(latent_dim,)))
        regressor.add(Dense(units=1))
        # Compile and train the regression model
        regressor.compile(optimizer='SGD', loss='mean_squared_error')
        regressor.fit(encoded_train, y_train, epochs=100, batch_size=32,verbose=0)
        # Make predictions on the test data
        predictions_self_val = regressor.predict(encoded_val)
        predictions_self_test = regressor.predict(encoded_test)
        # Convert predictions to 1D array if needed
        predictions = np.ravel(predictions_self_test)
        # Calculate the R-squared score
        ss_score = r2_score(y_test, predictions)

        #***************************************************************************************************************

        #  ensemble regression approach late fusion
        start = time()
        # Combine the predictions at the feature level
        combined_features = np.column_stack((y_pred_rigr, y_pred_sgdr, y_pred_gbr, y_pred_lr, y_pred_rndr,predictions_self_val))
        combined_features_test = np.column_stack((y_pred_rigr_test, y_pred_sgdr_test, y_pred_gbr_test, y_pred_lr_test, y_pred_rndr_test,predictions_self_test))
        # fusion_model = HyperoptEstimator(regressor=any_regressor('reg'), preprocessing=any_preprocessing('pre'),
        #                                loss_fn=mean_absolute_error, algo=tpe.suggest, max_evals=50, trial_timeout=30)
        fusion_model = GradientBoostingRegressor(**gbr_params)
        # fusion_model=SVR(kernel='linear')
        fusion_model.fit(combined_features, y_val)
        ems_predict = fusion_model.predict(combined_features_test)
        ems_score = fusion_model.score(combined_features_test, y_test)
        end = time()
        result = end - start
        print('%.3f seconds' % result)

        # Confusion matrix for ensemble model
        cm_ens = confusion_matrix(y_test, np.round(ems_predict))
        print("Confusion Matrix - Ensemble Model:")
        print(cm_ens)
        print()
        cm_self = confusion_matrix(y_test, np.round(predictions_self_test))
        print("Confusion Matrix - Self S. Model:")
        print(cm_self)
        print()
        # Confusion matrix for Random Forest model
        cm_rndr = confusion_matrix(y_test, np.round(y_pred_rndr_test))
        # Confusion matrix for SVR model
        cm_svr = confusion_matrix(y_test, np.round(y_pred_lr_test))
        # Confusion matrix for Ridge model
        cm_rigr = confusion_matrix(y_test, np.round(y_pred_rigr_test))
        # Confusion matrix for GBR model
        cm_gbr = confusion_matrix(y_test, np.round(y_pred_gbr_test))
        # Confusion matrix for SGDRegressor model
        #cm_sgdr = confusion_matrix(y_test, np.round(y_pred_sgdr))

        print("Confusion Matrix - Random Forest Model:")
        print(cm_rndr)
        print()

        print("Confusion Matrix - SVR Model:")
        print(cm_svr)
        print()

        print("Confusion Matrix - Ridge Model:")
        print(cm_rigr)
        print()

        print("Confusion Matrix - GBR Model:")
        print(cm_gbr)
        print()

        #print("Confusion Matrix - SGDRegressor Model:")
        #print(cm_sgdr)
        #print()

        print("R-squared:", "ENS", ems_score, "SS", ss_score, "RND : ", score_rndr, "SVR : ", score_lr, "rigr : ", score_rigr, "GBR : ",
              score_gbr, "SGDR : ",
              score_sgdr)
        print("MSE:",
              "ENS", mean_squared_error(y_test, ems_predict),
              "SS :",mean_squared_error(y_test, predictions_self_test),
              "RND : ", mean_squared_error(y_test, y_pred_rndr_test),
              "SVR : ", mean_squared_error(y_test, y_pred_lr_test),
              "rigr : ", mean_squared_error(y_test, y_pred_rigr_test),
              "GBR : ", mean_squared_error(y_test, y_pred_gbr_test),
              "SGDR : ", mean_squared_error(y_test, y_pred_sgdr_test))

        import scipy.stats

        # print("Pearson CCs",
        #      "Svr : ", scipy.stats.pearsonr(y_test, y_pred_rndr)[0])
        print("Pearson CCs",
              "ENS : ", scipy.stats.pearsonr(y_test, ems_predict)[0],
              "SS : ", scipy.stats.pearsonr(np.ravel(y_test), predictions)[0],
              "Rnd : ", scipy.stats.pearsonr(y_test, y_pred_rndr_test)[0],
              "svr : ", scipy.stats.pearsonr(y_test, y_pred_lr_test)[0],
              "rigr : ", scipy.stats.pearsonr(y_test, y_pred_rigr_test)[0],
              "gbr : ", scipy.stats.pearsonr(y_test, y_pred_gbr_test)[0],
              "sgdr : ", scipy.stats.pearsonr(y_test, y_pred_sgdr_test)[0])
        # print("Spearman CCs",
        #      "Svr : ", scipy.stats.spearmanr(y_test, y_pred_rndr)[0])
        print("Spearman CCs",
              "ENS : ", scipy.stats.spearmanr(y_test, ems_predict)[0],
              "SS : ", scipy.stats.spearmanr(np.ravel(y_test), predictions)[0],
              "Rnd : ", scipy.stats.spearmanr(y_test, y_pred_rndr_test)[0],
              "svr : ", scipy.stats.spearmanr(y_test, y_pred_lr_test)[0],
              "rigr : ", scipy.stats.spearmanr(y_test, y_pred_rigr_test)[0],
              "gbr : ", scipy.stats.spearmanr(y_test, y_pred_gbr_test)[0],
              "sgdr : ", scipy.stats.spearmanr(y_test, y_pred_sgdr_test)[0])

        ens.append(ems_score)
        ss.append(ss_score)
        rnd.append(score_rndr)
        svm.append(score_lr)
        rig.append(score_rigr)
        gebr.append(score_gbr)
        segdr.append(score_sgdr)

        if (scipy.stats.pearsonr(y_test, ems_predict)[0] > ens_f[0]):
            ens_f[0] = scipy.stats.pearsonr(y_test, ems_predict)[0]
            ens_feature[0] = features.features
            ens_p[0] = ems_score
            ens_pearson[0] = scipy.stats.pearsonr(y_test, ems_predict)[0]
            ens_spearman[0] = scipy.stats.spearmanr(y_test, ems_predict)[0]
            ens_kendal[0] = scipy.stats.kendalltau(y_test, ems_predict)[0]
            ens_mse[0] = mean_squared_error(y_test, ems_predict)
            ens_mae[0] = round(mean_absolute_error(y_test, ems_predict), 4)
            ens_expvar[0] = round(explained_variance_score(y_test, ems_predict), 4)

        if (scipy.stats.pearsonr(np.ravel(y_test), predictions)[0] > ss_f[0]):
            ss_f[0] = scipy.stats.pearsonr(np.ravel(y_test), predictions)[0]
            ss_feature[0] = features.features
            ss_p[0] = ss_score
            ss_pearson[0] = scipy.stats.pearsonr(np.ravel(y_test), predictions)[0]
            ss_spearman[0] = scipy.stats.spearmanr(np.ravel(y_test), predictions)[0]
            ss_kendal[0] = scipy.stats.kendalltau(np.ravel(y_test), predictions)[0]
            ss_mse[0] = mean_squared_error(np.ravel(y_test), predictions)
            ss_mae[0] = round(mean_absolute_error(np.ravel(y_test), predictions), 4)
            ss_expvar[0] = round(explained_variance_score(np.ravel(y_test), predictions), 4)

        if (scipy.stats.pearsonr(y_test, y_pred_rndr_test)[0] > rnd_f[0]):
            rnd_f[0] = scipy.stats.pearsonr(y_test, y_pred_rndr_test)[0]
            rnd_feature[0] = features.features
            rnd_p[0] = score_rndr
            rnd_pearson[0] = scipy.stats.pearsonr(y_test, y_pred_rndr_test)[0]
            rnd_spearman[0] = scipy.stats.spearmanr(y_test, y_pred_rndr_test)[0]
            rnd_kendal[0] = scipy.stats.kendalltau(y_test, y_pred_rndr_test)[0]
            rnd_mse[0] = mean_squared_error(y_test, y_pred_rndr_test)
            rnd_mae[0] = round(mean_absolute_error(y_test, y_pred_rndr_test), 4)
            rnd_expvar[0] = round(explained_variance_score(y_test, y_pred_rndr_test), 4)

        if (scipy.stats.pearsonr(y_test, y_pred_lr_test)[0] > svm_f[0]):
            svm_f[0] = scipy.stats.pearsonr(y_test, y_pred_lr_test)[0]
            svm_feature[0] = features.features
            svm_p[0] = score_lr
            svm_pearson[0] = scipy.stats.pearsonr(y_test, y_pred_lr_test)[0]
            svm_spearman[0] = scipy.stats.spearmanr(y_test, y_pred_lr_test)[0]
            svm_kendal[0] = scipy.stats.kendalltau(y_test, y_pred_lr_test)[0]
            svm_mse[0] = mean_squared_error(y_test, y_pred_lr_test)
            svm_mae[0] = round(mean_absolute_error(y_test, y_pred_lr_test), 4)
            svm_expvar[0] = round(explained_variance_score(y_test, y_pred_lr_test), 4)

        if (scipy.stats.pearsonr(y_test, y_pred_rigr_test)[0] > rig_f[0]):
            rig_f[0] = scipy.stats.pearsonr(y_test, y_pred_rigr_test)[0]
            rig_feature[0] = features.features
            rig_p[0] = score_rigr
            rig_pearson[0] = scipy.stats.pearsonr(y_test, y_pred_rigr_test)[0]
            rig_spearman[0] = scipy.stats.spearmanr(y_test, y_pred_rigr_test)[0]
            rig_kendal[0] = scipy.stats.kendalltau(y_test, y_pred_rigr_test)[0]
            rig_mse[0] = mean_squared_error(y_test, y_pred_rigr_test)
            rig_mae[0] = round(mean_absolute_error(y_test, y_pred_rigr_test), 4)
            rig_expvar[0] = round(explained_variance_score(y_test, y_pred_rigr_test), 4)

        if (scipy.stats.pearsonr(y_test, y_pred_gbr_test)[0] > gebr_f[0]):
            gebr_f[0] = scipy.stats.pearsonr(y_test, y_pred_gbr_test)[0]
            gebr_feature[0] = features.features
            gebr_p[0] = score_gbr
            gebr_pearson[0] = scipy.stats.pearsonr(y_test, y_pred_gbr_test)[0]
            gebr_spearman[0] = scipy.stats.spearmanr(y_test, y_pred_gbr_test)[0]
            gebr_kendal[0] = scipy.stats.kendalltau(y_test, y_pred_gbr_test)[0]
            gebr_mse[0] = mean_squared_error(y_test, y_pred_gbr_test)
            gebr_mae[0] = round(mean_absolute_error(y_test, y_pred_gbr_test), 4)
            gebr_expvar[0] = round(explained_variance_score(y_test, y_pred_gbr_test), 4)

        if (scipy.stats.pearsonr(y_test, y_pred_sgdr_test)[0] > segdr_f[0]):
            segdr_f[0] = scipy.stats.pearsonr(y_test, y_pred_sgdr_test)[0]
            segdr_feature[0] = features.features
            segdr_p[0] = score_sgdr
            segdr_pearson[0] = scipy.stats.pearsonr(y_test, y_pred_sgdr_test)[0]
            segdr_spearman[0] = scipy.stats.spearmanr(y_test, y_pred_sgdr_test)[0]
            segdr_kendal[0] = scipy.stats.kendalltau(y_test, y_pred_sgdr_test)[0]
            segdr_mse[0] = mean_squared_error(y_test, y_pred_sgdr_test)
            segdr_mae[0] = round(mean_absolute_error(y_test, y_pred_sgdr_test), 4)
            segdr_expvar[0] = round(explained_variance_score(y_test, y_pred_sgdr_test), 4)

        print("Kendalltau CCs",
              "ENS : ", scipy.stats.kendalltau(y_test, ems_predict)[0],
              "SS : ", scipy.stats.kendalltau(np.ravel(y_test), predictions)[0],
              "Rnd : ", scipy.stats.kendalltau(y_test, y_pred_rndr_test)[0],
              "svr : ", scipy.stats.kendalltau(y_test, y_pred_lr_test)[0],
              "rigr : ", scipy.stats.kendalltau(y_test, y_pred_rigr_test)[0],
              "gbr : ", scipy.stats.kendalltau(y_test, y_pred_gbr_test)[0],
              "sgdr : ", scipy.stats.kendalltau(y_test, y_pred_sgdr_test)[0])

        print("MAE =", "ENS", round(mean_absolute_error(y_test, ems_predict), 4),
              "SS : ", round(mean_absolute_error(np.ravel(y_test), predictions), 4),
              "Rnd :", round(mean_absolute_error(y_test, y_pred_rndr_test), 4), "svr :",
              round(mean_absolute_error(y_test, y_pred_lr_test), 4),
              "rigr :", round(mean_absolute_error(y_test, y_pred_rigr_test), 4), "gbr :",
              round(mean_absolute_error(y_test, y_pred_gbr_test), 4),
              "sgdr :", round(mean_absolute_error(y_test, y_pred_sgdr_test), 4))

        print("MedAE =", "ENS", round(median_absolute_error(y_test, ems_predict), 4),
              "SS : ", round(median_absolute_error(np.ravel(y_test), predictions), 4),
              "Rnd :", round(median_absolute_error(y_test, y_pred_rndr_test), 4), "svr :",
              round(median_absolute_error(y_test, y_pred_lr_test), 4),
              "rigr :", round(median_absolute_error(y_test, y_pred_rigr_test), 4), "gbr :",
              round(median_absolute_error(y_test, y_pred_gbr_test), 4),
              "sgdr :", round(median_absolute_error(y_test, y_pred_sgdr_test), 4))

        print("ExpVarSc =", "ENS", round(explained_variance_score(y_test, ems_predict), 4),
              "SS : ", round(explained_variance_score(np.ravel(y_test), predictions), 4),
              "Rnd :", round(explained_variance_score(y_test, y_pred_rndr_test), 4), "svr :",
              round(explained_variance_score(y_test, y_pred_lr_test), 4),
              "rigr :", round(explained_variance_score(y_test, y_pred_rigr_test), 4), "gbr :",
              round(explained_variance_score(y_test, y_pred_gbr_test), 4),
              "sgdr :", round(explained_variance_score(y_test, y_pred_sgdr_test), 4))

        # print("Balanced Acc =","Rnd :", (balanced_accuracy_score(y_test, y_pred_rndr), 4),"svr :", (balanced_accuracy_score(y_test, y_pred_lr), 4),
        #     "rigr :", (balanced_accuracy_score(y_test, y_pred_rigr), 4),"gbr :", (balanced_accuracy_score(y_test, y_pred_gbr), 4),
        #    "sgdr :", (balanced_accuracy_score(y_test, y_pred_sgdr), 4))
        print("R2 =", "ENS :", round(r2_score(y_test, ems_predict), 4),
              "SS : ", round(r2_score(np.ravel(y_test), predictions), 4),
              "Rnd :", round(r2_score(y_test, y_pred_rndr_test), 4),
              "svr :", round(r2_score(y_test, y_pred_lr_test), 4),
              "rigr :", round(r2_score(y_test, y_pred_rigr_test), 4),
              "gbr :", round(r2_score(y_test, y_pred_gbr_test), 4),
              "sgdr :", round(r2_score(y_test, y_pred_sgdr_test), 4))


    def Average(lst):
        return sum(lst) / len(lst)


    print("R2 Best rnd:", rnd_f[0], "ENS", ens_f[0],"SS", ss_f[0], "svm:", svm_f[0], "rig:",
          rig_f[0], "gebr:", gebr_f[0], "segdr:", segdr_f[0])
    print("Pearson Best rnd:", rnd_pearson[0], "ENS", ens_pearson[0],"SS", ss_pearson[0], "svm:", svm_pearson[0], "rig:",
          rig_pearson[0], "gebr:", gebr_pearson[0], "segdr:", segdr_pearson[0])
    print("Spearman rnd:", rnd_spearman[0], "ENS", ens_spearman[0],"SS", ss_spearman[0], "svm:", svm_spearman[0], "rig:",
          rig_spearman[0], "gebr:", gebr_spearman[0], "segdr:", segdr_spearman[0])
    print("Kendall Best rnd:", rnd_kendal[0], "ENS", ens_kendal[0],"SS", ss_kendal[0], "svm:", svm_kendal[0], "rig:",
          rig_kendal[0], "gebr:", gebr_kendal[0], "segdr:", segdr_kendal[0])
    print("expVar rnd:", rnd_expvar[0], "ENS", ens_expvar[0],"SS", ss_expvar[0], "svm:", svm_expvar[0], "rig:",
          rig_expvar[0], "gebr:", gebr_expvar[0], "segdr:", segdr_expvar[0])
    print("MSE rnd:", rnd_mse[0], "ENS", ens_mse[0],"SS", ss_mse[0], "svm:", svm_mse[0], "rig:",
          rig_mse[0], "gebr:", gebr_mse[0], "segdr:", segdr_mse[0])
    print("MAE rnd:", rnd_mae[0], "ENS", ens_mae[0],"SS", ss_mae[0], "svm:", svm_mae[0], "rig:",
          rig_mae[0], "gebr:", gebr_mae[0], "segdr:", segdr_mae[0])
    print("expVar rnd:", rnd_feature[0],"\n","ENS", ens_feature[0],"\n","SS", ss_feature[0],"\n", "svm:", svm_feature[0],"\n", "rig:",
          rig_feature[0],"\n", "gebr:", gebr_feature[0],"\n", "segdr:", segdr_feature[0])
    # Convert the correlation coefficient values to NumPy arrays
    # Convert the correlation coefficient values to NumPy arrays
    pcc_values = np.array(pcc_values)
    scc_values = np.array(scc_values)
    kcc_values = np.array(kcc_values)


    # Bland-Altman Plot Function
    def bland_altman_plot(data1, data2, label1, label2, correlation_type):
        mean_values = np.mean([data1, data2], axis=0)
        diff = data1 - data2
        mean_diff = np.mean(diff)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(mean_values, diff, label=f'{correlation_type} Difference')
        ax.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')

        # Adding labels
        ax.set_xlabel(f'Mean of {label1} and {label2}')
        ax.set_ylabel(f'{correlation_type} Difference')
        ax.set_title(f'Bland-Altman Plot for {correlation_type}')
        ax.legend()

        plt.show()


    # Create Bland-Altman plots for each correlation coefficient
    bland_altman_plot(pcc_values, scc_values, 'PCC', 'SCC', 'Pearson vs. Spearman')
    bland_altman_plot(pcc_values, kcc_values, 'PCC', 'KCC', 'Pearson vs. Kendall')
    bland_altman_plot(scc_values, kcc_values, 'SCC', 'KCC', 'Spearman vs. Kendall')