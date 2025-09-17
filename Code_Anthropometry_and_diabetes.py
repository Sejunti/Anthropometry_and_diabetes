#!/usr/bin/env python
# coding: utf-8

# ## Import libaries 

# In[1]:


import pandas as pd
import networkx as nx
get_ipython().run_line_magic('pylab', 'inline')
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
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt


# ## Function for import data and merge the datasets 

# In[2]:


def import_data(filename):
    """
    This function is used how to import a dataset from a file.
    """
    data=pd.read_csv(filename)
    return data
def drop_rows(df, n):
    """
    This is the function for remove rows from a dataframe. 
    """
    return df.iloc[:-n] if n < len(df) else pd.DataFrame(columns=df.columns)


# In[3]:


filename='Anthropometry_and_diabetes.csv'
data= pd.read_csv(filename)


# ## Train and test data

# In[4]:


X=data[['age2018','Weight_kg','Height_cm','BMI','WholeBodyFatPerc','TrunkPerc','meangripboth','generate waistc','generate waisthp','generate waistht','generate_hipc']]


# ##  Testing data (y) is changed according to the target.

# In[5]:


y=data['dibaHbA1C'] # For Diabetes
y=data['htnbin'] # For Hypertension
y=data['binary_ability']  # For ability to work


# In[6]:


scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=16)


# ## Function for finding participant characteristics (mean, median, standard deviation, maximum, minimum)

# In[7]:


def basic_property(X):
    '''
    This function is used to compute the characteristics such as mean,median, standard deviation, maximum, and minimum
    of the participant from the dataset.
    '''
  
    mean=X.mean() #mean
    median=X.median() #median
    stndd=X.std()#standard deviation
    maximum=X.max() # maximum
    minimum=X.min() #minimum 
    return mean, median, stndd,maximum,minimum


# ## Decision tree classification  

# In[8]:


def decisiontreeclassification(X_train, y_train):
    '''
    This function is used to find the feature importance of the variable by decision tree classifier.
    '''
    
    clf=DecisionTreeClassifier(random_state = 42, class_weight="balanced")
    output = cross_validate(clf, X_train, y_train,cv=3, scoring = 'accuracy', return_estimator =True)
    for idx, estimator in enumerate(output['estimator']):
        feature_importances = pd.DataFrame(estimator.feature_importances_,index=X.columns,  # Assuming X is a DataFrame
                                       columns=['importance']).sort_values('importance', ascending=False)
    return feature_importances

    


# ## Logistic regression models 

# In[9]:


def logisticregresssionmodel(X_train, y_train):
    '''
    This function is used to find the coefficient value for each feature by the logistic refression model.
    '''
    mod_log = OrderedModel(y_train,X_train,distr='logit')
    res_log = mod_log.fit(method='bfgs', disp=False)
    return res_log.summary()


# ## Percentile figure plot  for Diabetes

# In[ ]:


data_Fd=data[data['dibaHbA1C']==1]
data_Fnd =data[data['dibaHbA1C']==0]
percentiles = np.linspace(0,1,1000)
r1 = data_Fd['generate waistc'].quantile(percentiles)
r2 = data_Fnd['generate waistc'].quantile(percentiles)
plt.plot(percentiles,r1)
plt.plot(percentiles,r2)
legend(['Diabetic Female','Non-diabetic Female'],fontsize=18)
plt.xlabel('Percentiles',fontsize=18)
plt.ylabel('Freuency',fontsize=18)


# ## Percentile plot for Hypertension

# In[ ]:


data_Fd =data[data['htnbin']==1]
data_Fnd =data[data['htnbin']==0]
percentiles = np.linspace(0,1,1000)
r1 = data_Fd['WholeBodyFatPerc'].quantile(percentiles)
r2 = data_Fnd['WholeBodyFatPerc'].quantile(percentiles)
plt.plot(percentiles,r1)
plt.plot(percentiles,r2)
legend(['Hypertension Female','Non-hypertension Female'],fontsize=18,loc='upper right')
plt.xlabel('Percentiles',fontsize=18)
plt.ylabel('Freuency',fontsize=18)


# ## Percentile Plot for ability to work

# In[ ]:


data_Fd =data[data['binary_ability']==1]
data_Fnd =data[data['binary_ability']==0]
percentiles = np.linspace(0,1,1000)
r1 = data_Fd['meangripboth'].quantile(percentiles)
r2 = data_Fnd['meangripboth'].quantile(percentiles)
plt.plot(percentiles,r1)
plt.plot(percentiles,r2)
legend(['Female who are able to do work','Female who are not able to do work'],fontsize=14,loc='upper right')
plt.xlabel('Percentiles',fontsize=18)
plt.ylabel('Freuency',fontsize=18)


# ## ROC curve plot code

# In[ ]:


logreg = LogisticRegression(max_iter=1000)
model_score=logreg.fit(X_train, y_train)
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.ylabel('True Positive Rate',fontsize=18)
plt.xlabel('False Positive Rate',fontsize=18)
plt.legend()
plt.show()


# ##  Waist circumference sensitivity and specificity for diabetes across potential screening cutoff values

# In[ ]:


subset_df = data[(data['generate waistc'] >= 70) & (data['generate waistc'] <= 100)]
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


plt.figure(figsize=(10, 6))
plt.plot(thresholds, sensitivity_values, label='Sensitivity')
plt.plot(thresholds, specificity_values, label='Specificity')
plt.xlabel('Waist circumference cutoff value',fontsize=18)
plt.ylabel('Sensitivity/Specifity',fontsize=18)
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




