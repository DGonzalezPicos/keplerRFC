"""
Created on Wed Nov 11 15:58:00 2020
@author: dariogp


"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def RFC(file, feat):
    data = pd.read_csv(file)
    y = data.koi_disposition
    featclean = []
    featclean.append('rowid')   
    for k in data.columns:
        if k in feat:
            featclean.append(k)
    
    X = data[featclean]
    X = X.dropna(axis=0)
    missing = []
    for k in range(1,len(y)):
        if (k in X.rowid) is False:
            missing.append(k)
        
    y = y.drop(missing)
    X = X.drop(columns='rowid')
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    forest_model = RandomForestClassifier(random_state=1, n_estimators=100)
    forest_model.fit(train_X, train_y)
    print("Training model...")   
    accuracy = forest_model.score(train_X, train_y), forest_model.score(val_X, val_y)
    print('Accuracy on the training {:.2f} and the validation data {:.2f}'.format(*accuracy))
    
#   Visualizing Feature importances
    importances = forest_model.feature_importances_
    indices = np.argsort(importances)
    feature_names = list(train_X.columns)
    xlabels = [feature_names[i] for i in indices] 
    y_ticks = np.arange(0, len(feature_names))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.barh(y_ticks, importances[indices])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(xlabels)
    
    ax.set_title("Random Forest Feature Importances (MDI)")
    plt.show()
    return forest_model
    
file = "./keplerRFC/cumulative.csv"
feat = ['koi_prad', 'koi_period', 'koi_impact', 'koi_duration']
RFC(file, feat)
