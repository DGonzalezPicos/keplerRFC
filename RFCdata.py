"""
Created on Fri Nov 13 10:33:05 2020
@author: dariogp

Support script for RFC data preparation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def correlation(train_X, feat, eps=0.70):
    '''
    Calculate the correlation among variables with the Pearson method
    INPUT: training data, list of features
    OUTPUT: correlation matrix, list with pairs of correlated variables, heatmap fig
    '''    
    corr_df = train_X[feat]  #New dataframe to calculate correlation between numeric features
    cor = corr_df.corr(method='pearson')
#    print(cor)
#    print('----------------------------------------------------------------------')
    correlated = []
    print("Highly correlated features:")

    for i in feat:
        for j in feat:
            eps1 = cor[i][j]
            if i != j:
                if eps1 > eps:   # arbitrary threshold
                    if ([j,i] in correlated)==True: # avoid adding the same pair twice
                        None
                    else:
                            correlated.append([i,j])
                            print('{:15s} & {:15s} \t R2 = {:.2f}'.format(i,j, eps1))
    
    fig, ax =plt.subplots(figsize=(8, 6))
    plt.title("Correlation Plot")
    
    sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), 
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.tight_layout()
    try:
        plt.savefig('./images/pearson.png', dpi=150)
    except:
        print('The directory ./images does not exist! Unable to save pearson.png')
    plt.show()
    return cor, correlated, fig
    
