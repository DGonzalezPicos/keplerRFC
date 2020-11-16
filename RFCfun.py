"""
Created on Wed Nov 11 15:58:00 2020
@author: dariogp

Main script for RFC
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#%%
file = "./koicumulative.csv"
data = pd.read_csv(file, sep=",")
feat = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
       'koi_period', 'koi_time0bk',
       'koi_time0', 'koi_impact', 'koi_duration',
       'koi_depth', 'koi_ror', 'koi_srho', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq', 'koi_insol', 'koi_dor',
       'koi_ldm_coeff2', 'koi_ldm_coeff1', 'koi_max_sngle_ev',
       'koi_max_mult_ev', 'koi_model_snr', 'koi_count', 'koi_num_transits',
       'koi_tce_plnt_num', 'koi_quarters',
       'koi_bin_oedp_sig','koi_steff', 'koi_slogg',
       'koi_smet', 'koi_srad', 'koi_smass','ra',
       'dec', 'koi_kepmag', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag',
       'koi_jmag', 'koi_hmag', 'koi_kmag', 'koi_fwm_sra',
       'koi_fwm_sdec', 'koi_fwm_srao', 'koi_fwm_sdeco', 'koi_fwm_prao',
       'koi_fwm_pdeco', 'koi_dicco_mra', 'koi_dicco_mdec', 'koi_dicco_msky',
       'koi_dikco_mra', 'koi_dikco_mdec', 'koi_dikco_msky']
      
       
def RFC(file, feat):
    data = pd.read_csv(file)
    y = data.koi_disposition
    # Clean data
#    featclean = []
    feat.append('rowid')   
#    for k in data.columns:
#        if k in feat:
#            featclean.append(k)
    
    X = data[feat]
    X = X.dropna(axis=0)
    missing = []
    for k in range(1,len(y)):
        if (k in X.rowid) is False:
            missing.append(k)
        
    y = y.drop(missing)
    X = X.drop(columns='rowid')
    # Build model
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.7, random_state=0)
    forest_model = RandomForestClassifier(random_state=1, 
                                          n_estimators=500,
                                          max_depth=16,
                                          max_features='auto')
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
    fig, ax = plt.subplots(figsize=(14,12))
    ax.barh(y_ticks, importances[indices])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(xlabels)
    
    ax.set_title("Random Forest Feature Importances (MDI)")
    plt.savefig('images/importance.png', dpi=150)
    plt.show()
    return train_X, val_X, train_y, val_y, xlabels, forest_model
    
#train_X, val_X, train_y, val_y, importances, rfcmodel = RFC(file, feat)

# =============================================================================
#              Correlation between numerical features in the dataset
# =============================================================================
from RFCdata import correlation
#==============================================================================
#                            MORE FEATURES
#==============================================================================
#transit_feat = ['koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad',
#                'koi_teq', 'koi_kepmag']  

print('Building model...')              
train_X, val_X, train_y, val_y, importances, rfcmodel = RFC(file, feat)   
print('Computing correlations...')     
cor, correlated, corfig = correlation(train_X, train_X.columns, eps=0.90)

#%%
#==============================================================================
#           Eliminate correlated features
#==============================================================================
#importances = importances[::-1] # natural order
#print(importances)
#for j in importances:
#    for i in importances:
#        if ([j,i] in correlated) == True:
#            if importances.index(j) > importances.index(i):
#                importances.remove(i)
#            if importances.index(j) < importances.index(i):
#                importances.remove(j)
#%%
# =============================================================================
#                   Parameter selection with GridSearchCV
# =============================================================================
#import datetime
#begin_time = datetime.datetime.now()
#print('Starting GridSearch...', begin_time)
#
# 
#from sklearn.model_selection import GridSearchCV
# 
#parameters = {'n_estimators': np.arange(100,601,100), 
#           'max_features': [None, 'auto','sqrt','log2'],
#           'max_depth': [6,10,12,16]}
#clf = GridSearchCV(rfcmodel, parameters)
#clf.fit(train_X,train_y)            
#print('Best parameters... ',clf.best_params_)
#print('Best score = {:.2f}'.format(clf.best_score_))
# 
#print('Execution time...', datetime.datetime.now() - begin_time)

# 13/11 at 14h
# Best parameters...  {'max_depth': 16, 'n_estimators': 500, 'max_features': 'auto'}
# Execution time... 0:13:26.552741


#%% 
# =============================================================================
#                   PREDICT OBJECT CLASS
# =============================================================================

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = rfcmodel.predict(val_X)
cm = confusion_matrix(val_y, y_pred)

row_sums = cm.sum(axis=1)
cm_normed = cm / row_sums[:, np.newaxis]

names = val_y.unique()
print(names)
fig, axs = plt.subplots(ncols=2, figsize=(16,6))
for matrix, style, axes in [(cm,'d',0), (cm_normed,'.2%', 1)]:
    sns.heatmap(matrix, xticklabels=names, yticklabels=names, annot=True, fmt=style, cmap='Blues',ax=axs[axes])

plt.savefig('images/confmatrix.png', dpi=150)
plt.show()

