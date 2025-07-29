#!/usr/bin/env python
# coding: utf-8

# ## Import libaries 

# In[1]:


import pandas as pd
import networkx as nx
import numpy as np
from random import *
from numpy import *
from random import *
import numpy.linalg as la
from scipy.linalg import eig
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, f1_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ## Formula for measuring some properties

# In[ ]:


data['hip_circumference'] = data['hipccm'] + (data['Hip_8ths'])/8 
data['waist_circumference'] = data['waistc'] *2.54
data['waist_to_height_ratio'] = data['waistccm']/ data['Height_cm']
data['waist_to_hip_ratio'] = data['waistccm']/ data['hipccm']


# ## Function for finding participant characteristics (mean, median, standard deviation, maximum, minimum)

# In[2]:


def basic_property(X):
    ''''
    This function is used to compute the characteristics such as mean,median, standard deviation, maximum, and minimum
    of the participant from the dataset.
    ''''
    mean=X.mean() #mean
    median=X.median() #median
    stndd=X.std()#standard deviation
    maximum=X.max() # maximum
    minimum=X.min() #minimum 
    return mean, median, stndd,maximum,minimum


# ## Train and test data

# In[ ]:


scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=16)


# ## Decision tree classification  

# In[6]:


def decisiontreeclassification(X_train, y_train):
    ''''
    This function is used to find the feature importance from the dataset. We used Decision tree classifier 
    method.
    ''''
    clf=DecisionTreeClassifier(random_state = 42, class_weight="balanced")
    output = cross_validate(clf, X_train, y_train,cv=1, scoring = 'accuracy', return_estimator =True)
    for idx, estimator in enumerate(output['estimator']):
        feature_importances = pd.DataFrame(estimator.feature_importances_,index=X.columns,  # Assuming X is a DataFrame
                                       columns=['importance']).sort_values('importance', ascending=False)
    return feature_importances

    


# ## Logistic regression models 

# In[7]:


def logisticregresssionmodel(X_train, y_train):
    ''''
    This function is used to find the coefficient value for each feature by the logistic refression model.
    ''''
    mod_log = OrderedModel(y_train,X_train,distr='logit')
    res_log = mod_log.fit(method='bfgs', disp=False)
    return res_log.summary()


# ## Percentile figure plot function

# In[ ]:


def percentile_graph('parameter'):
    '''
    This function is used to plot the property percentile amonf women with and without diabetes, 
    hypertensio and able to work.
    '''
    percentiles = np.linspace(0,1,1000)
    r1 = data_Fd['parameter'].quantile(percentiles) # Dataset who have diabetes, hypertension, able to work
    r2 = data_Fnd['parameter'].quantile(percentiles)# Dataset who don't have diabetes, hypertension, able to work
    return r1,r2
    


# ## ROC curve plot code

# In[ ]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')


# ##  Waist circumference sensitivity and specificity for diabetes across potential screening cutoff values

# In[ ]:


subset_df = data[(data['waist_circumference'] >= 70) & (data['waist_circumference'] <= 100)]
# Calculate sensitivity and specificity
sensitivity_values = []
specificity_values = []
thresholds = []

for threshold in range(70, 101):
    threshold_df = subset_df.copy()
    threshold_df['predicted_diabetes'] = (threshold_df['generate waistc'] >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(threshold_df['dibaHbA1C'], threshold_df['predicted_diabetes']).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_values.append(sensitivity)
    specificity_values.append(specificity)
    thresholds.append(threshold)

plt.plot(thresholds, sensitivity_values, label='Sensitivity')
plt.plot(thresholds, specificity_values, label='Specificity')

