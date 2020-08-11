#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 10 2020
@author: Georgia
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
       'feature_importance': importances}) \
        .sort_values('feature_importance', ascending = False) \
        .reset_index(drop = True)
    return df

# eli5 feature importance
def permu_importance(model, X_test,y_test, feature_names, n_iter = 5):
    #refit the model on test set
    perm_MSE = PermutationImportance(model,scoring = 'neg_mean_squared_error', random_state= 42, n_iter=n_iter).fit(X_test, y_test)

    PI_mean = imp_df(feature_names, perm_MSE.feature_importances_)
    print('MSE',PI_mean)

    PI_std =imp_df(feature_names,perm_MSE.feature_importances_std_)
    print('PI_std',PI_std)

    return perm_MSE

def evaluate(model, X, y):
    """Evaluate the model and calculate the mean absolute error, root mean square error"""
    print("Evaluate the model and calculate the mean absolute error, Root mean square error")
    predictions = model.predict(X)
    final_mae = mean_absolute_error(y, predictions)

    final_mse = mean_squared_error(y, predictions)
    final_rmse = np.sqrt(final_mse)

    print('Mean Absolute Error:', round(final_mae,5))
    print('Root Mean Squarer Error:', round(final_rmse,5))

    return predictions, final_rmse,final_mae


def rf_base (X_train, y_train, X_test, y_test):
    "random forest based model"
    base_model = RandomForestRegressor(n_estimators=10, random_state=42)
    base_model.fit(X_train, y_train)
    print("Base model- model performance:")
    base_pred, base_rmse,base_mae = evaluate(base_model, X_test, y_test)
    return base_rmse,base_mae

def randomsearch(model,random_grid,X_train,y_train):
    "performs random search, for RF and SVM"
    rand_model = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               scoring='neg_mean_absolute_error',random_state=42, n_jobs=-1)
    rand_model.fit(X_train, y_train)
    print("random serach best hyperparameters", rand_model.best_params_)
    return rand_model


def gridsearch(model,grid_grid,X_train,y_train):
    "performs grid search, for RF and SVM"
    # Instantiate the grid search model
    grid_search_model = GridSearchCV(estimator=model, param_grid=grid_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

    # Fit the grid search to the data
    grid_search_model.fit(X_train, y_train)
    print(grid_search_model.best_params_)

    return grid_search_model

def xgb_grid(dtrain,params,gridsearch_params,num_boost_round):
    "grid search for XGBOOST"
    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth, min_child_weight)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    return


#plotting random forest
def plot_RF_mse(y,X):
    #TODO change this, according to the already trained rf_grid model, use mean sqaure error and mean abbsolute error,train set error
    print("Plot RF training..")
    estimators= [2,10,30,50,100, 200,300,600,800]
    print("estimators",estimators)
    mean_all = []
    std_upper = []
    std_lower = []
    yt = [i for i in y]
    np.random.seed(11111)
    for i in estimators:
        model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=56,
                              max_features='auto', max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=3, min_samples_split=3,
                              min_weight_fraction_leaf=0.0, n_estimators=i,
                              n_jobs=-1, oob_score=False, random_state=None,
                              verbose=0, warm_start=False)

        mse_scores= cross_val_score(model,X,yt, cv=3, scoring='neg_mean_squared_error')
        #rmse_scores = np.sqrt(-mse_scores)
        mse_scores=-mse_scores
        print('estimators:',i)
    #   print('explained variance scores for k=10 fold validation:',scores_rfr)
        #print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
        #print("MAE score: %0.5f (+/- %0.5f)" % (mbs_scores.mean(), mbs_scores.std() * 2))
        print("MSE score: %0.5f (+/- %0.5f)" % (mse_scores .mean(), mse_scores.std() * 2))
        # mean_all.append(mbs_scores.mean())
        # std_upper.append(mbs_scores.mean() + mbs_scores.std() * 2)  # for error plotting
        # std_lower.append(mbs_scores.mean() - mbs_scores.std() * 2)  # for error plotting

        mean_all.append(mse_scores.mean())
        std_upper.append(mse_scores.mean()+mse_scores.std()*2) # for error plotting
        std_lower.append(mse_scores.mean()-mse_scores.std()*2) # for error plotting
    # plot the figure
    fig = plt.figure(figsize=(12,8))
    csfont = {'fontname':'DejaVu Sans'}
    ax = fig.add_subplot(111)
    ax.plot(estimators,mean_all,marker='o',
           linewidth=4,markersize=12)
    ax.fill_between(estimators,std_lower,std_upper,
                    facecolor='green',alpha=0.3,interpolate=True)
    #ax.set_ylim([0.3,0.8])
    #ax.set_xlim([0,300])
    #ax.set_xticklabels(x_ticks, rotation=0, fontsize=8)
    #ax.set_yticklabels(y_ticks, rotation=0, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=15)
    #ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.tick_params(direction='out', length=5, width=2, colors='black',
                   grid_color='grey', grid_alpha=0.5)
    #plt.rc('xtick',labelsize=18)
    #plt.rc('ytick',labelsize=18)
    plt.title(' Mean Squared Errors of Random Forest',fontsize=20, fontweight='bold')
    #plt.title('mbs of Random Forest Regressor', fontsize=14, fontweight='bold')
    plt.ylabel('MSE',fontsize=20,**csfont)
    #plt.ylabel('MBS',fontsize=14)
    plt.xlabel('Estimator',fontsize=20,**csfont)
    plt.grid()
    plt.savefig('Random forest_MSE.png', dpi=300, bbox_inches="tight")

def plot_RF_mae(y, X):
    print("Plot RF training..")
    estimators= [2,10,30,50,100,200,300,600,800]
    print("estimators",estimators)
    mean_all = []
    std_upper = []
    std_lower = []

    yt = [i for i in y] # quick pre-processing of the target
    np.random.seed(11111)
    for i in estimators:
        model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=56,
                              max_features='auto', max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=3, min_samples_split=3,
                              min_weight_fraction_leaf=0.0, n_estimators=i,
                              n_jobs=-1, oob_score=False, random_state=None,
                              verbose=0, warm_start=False)

        mbs_scores= cross_val_score(model,X,yt, cv=3, scoring='neg_mean_absolute_error')
        mbs_scores=-mbs_scores
        print('estimators:',i)
        print("MAE score: %0.5f (+/- %0.5f)" % (mbs_scores.mean(), mbs_scores.std() * 2))
        mean_all.append(mbs_scores.mean())
        std_upper.append(mbs_scores.mean() + mbs_scores.std() * 2)  # for error plotting
        std_lower.append(mbs_scores.mean() - mbs_scores.std() * 2)  # for error plotting

    # plot the MAE figure
    fig = plt.figure(figsize=(12,8))
    csfont = {'fontname':'DejaVu Sans'}
    ax = fig.add_subplot(111)
    ax.plot(estimators,mean_all,marker='o',
           linewidth=4,markersize=12)
    ax.fill_between(estimators,std_lower,std_upper,
                    facecolor='green',alpha=0.3,interpolate=True)
    #ax.set_ylim([0.3,0.8])
    #ax.set_xlim([0,300])
    #ax.set_xticklabels(x_ticks, rotation=0, fontsize=8)
    #ax.set_yticklabels(y_ticks, rotation=0, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(direction='out', length=5, width=2, colors='black',
                   grid_color='grey', grid_alpha=0.5)
    #plt.rc('xtick',labelsize=18)
    #plt.rc('ytick',labelsize=18)
    plt.title('Mean Absolute Errors of Random Forest',fontsize=20, fontweight='bold')
    plt.ylabel('MAE',fontsize=20,**csfont)
    #plt.ylabel('MBS',fontsize=14)
    plt.xlabel('Estimator',fontsize=20,**csfont)
    plt.grid()
    plt.savefig('Random forest_MAE.png', dpi=300, bbox_inches="tight")
