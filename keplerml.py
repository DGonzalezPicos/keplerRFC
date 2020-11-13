"""
Created on Tue Nov 10 12:27:53 2020
@author: dariogp


"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#==============================================================================
#                           READ DATA
#==============================================================================
# import data
file = "./koicumulative.csv"
data = pd.read_csv(file)

# drop empty data points
#data = data.dropna(axis=0)
data.head()

# predicted target >>> koi_disposition
#y = data[['rowid', 'koi_disposition']]
y = data.koi_disposition
y.head()
# choose features
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
       'koi_jmag', 'koi_hmag', 'koi_kmag']
#       'koi_fwm_sra',
#       'koi_fwm_sdec', 'koi_fwm_srao', 'koi_fwm_sdeco', 'koi_fwm_prao',
#       'koi_fwm_pdeco', 'koi_dicco_mra', 'koi_dicco_mdec', 'koi_dicco_msky',
#       'koi_dikco_mra', 'koi_dikco_mdec', 'koi_dikco_msky']
# the given dataset has less features than the original kepler database
# remove the labels that are not in the dataset
#featclean = []
feat.append('rowid')
#for k in data.columns:
#    if k in feat:
#        featclean.append(k)

# data for the model
X = data[feat]
X.head()
# remove incomplete rows
X = X.dropna(axis=0)

missing = []
for k in range(1,len(y)):
    if (k in X.rowid) is False:
        missing.append(k)
    
y = y.drop(missing)
y.describe()

# drop the first row >> rowid
X = X.drop(columns='rowid')
X.columns
y[0]
#%%
#==============================================================================
#                           BUILD RFC MODEL
#==============================================================================
# Split the data into training and validation data
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(random_state=1, n_estimators=100)


forest_model.fit(train_X, train_y)

# Accuracy of the model 
accuracy = forest_model.score(train_X, train_y), forest_model.score(val_X, val_y)
print(accuracy)



data_preds = forest_model.predict(val_X)

# compare the model
acc = 0
for i,j in zip(data_preds, val_y):
    if i==j:
        acc += 1
        
print("The accuracy of the model is {:2.2f} %".format(100*acc/len(data_preds)))

#%%
#==============================================================================
#                   Feature importances with forest of trees
#==============================================================================

importances = forest_model.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest_model.estimators_], axis=0)
indices = np.argsort(importances)

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# features names and reordering
feature_names = list(train_X.columns)
xlabels = [feature_names[i] for i in indices] 

# Plot the impurity-based feature importances of the forest
y_ticks = np.arange(0, len(importances))
fig, ax = plt.subplots(figsize=(12,8))
ax.barh(y_ticks, importances[indices])
ax.set_yticks(y_ticks)
ax.set_yticklabels(xlabels)

ax.set_title("Random Forest Feature Importances (MDI)")
#fig.tight_layout()
plt.show()


#%%
#==============================================================================
#               Feature selection
#==============================================================================
# run again the RFC ONLY with the most relevant features
xlabelsrev = xlabels[::-1]
newfeat = xlabelsrev[:5]

# Build new model
X = data[newfeat]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
forest_model = RandomForestClassifier(random_state=1, n_estimators=100)
forest_model.fit(train_X, train_y)
accuracy = forest_model.score(train_X, train_y), forest_model.score(val_X, val_y)
print('Accuracy on the training {:.2f} and the validation data {:.2f}'.format(*accuracy))

#%%
# NEXT >>> Is accuracy calculated like that? Other statistics... recall, precision...
# Compare results to clayton.pdf
