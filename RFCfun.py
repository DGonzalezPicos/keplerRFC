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
    return train_X, val_X, train_y, val_y, forest_model
    
file = "./cumulative.csv"
feat = ['koi_prad', 'koi_period', 'koi_impact', 'koi_duration']
train_X, val_X, train_y, val_y, rfcmodel = RFC(file, feat)


#%%
# =============================================================================
#                   Parameter selection with GridSearchCV
# =============================================================================
import datetime
begin_time = datetime.datetime.now()

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': np.arange(100,601,100), 
              'max_features': [None, 'auto','sqrt','log2'],
              'max_depth': [6,10,12]}
clf = GridSearchCV(rfcmodel, parameters)
clf.fit(train_X,train_y)            
print('Best parameters... ',clf.best_params_)


print('Execution time...', datetime.datetime.now() - begin_time)
#%%
# =============================================================================
#              Correlation between numerical features in the dataset
# =============================================================================
import seaborn as sns
corr_df=X[feat]  #New dataframe to calculate correlation between numeric features
cor= corr_df.corr(method='pearson')
print(cor)
print('----------------------------------------------------------------------')
correlated = []
for i in feat:
    for j in feat:
        eps = cor[i][j]
        if i != j:
            if eps > 0.6:   # arbitrary threshold
                if ([j,i] in correlated)==True: # avoid adding the same pair twice
                    None
                else:
                        correlated.append([i,j])
                        print('The features {:s} and {:s} are highly correlated. R2 = {:.2f}'.format(i,j, eps))
#%%            
fig, ax =plt.subplots(figsize=(8, 6))
plt.title("Correlation Plot")

sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.savefig('../Images/pearson.png', dpi=150)
plt.show()
# there is significant correlation between koi_impact and koi_prad as expected
''' It seems that multicollinearity is well handled by RFC, however, it will make
it easier and faster for the user to eliminate highly correlated variables,
with pearson > 0.7 for instance '''


