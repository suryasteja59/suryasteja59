# -*- coding: utf-8 -*-
"""
Created on Fri May 28 03:17:17 2021

@author: Surya Teja
"""

##################################### WORK FLOW ################################################
            

        
#Step 1
# Importing essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Step 2
# Importing the given data (train and test)
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
data = pd.concat([train_data, test_data], axis=0)


#step 3
#Copying data into separate DataFrame
df = data.copy()



####################### Basic Plot Analysis ####################################
# Since most of the data is pretty straight forward we will be looking only few plots

# This is a imbalanced dataset
plt.figure()
sns.countplot(df['Is_Lead'])

# Almost equally distributed
plt.figure()
sns.countplot(df['Gender'], hue=df['Is_Lead'])

# No outliers
plt.figure()
sns.histplot(df['Age'])
plt.figure()
sns.boxplot(df['Age'])

# Entrepreneur more interested leads
plt.figure()
sns.countplot(df['Occupation'], hue=df['Is_Lead'])

# Less leads from channel 1
plt.figure()
sns.countplot(df['Channel_Code'], hue=df['Is_Lead'])

# No outliers (distribution similar to age)
plt.figure()
sns.histplot(df['Vintage'])
plt.figure()
sns.boxplot(df['Vintage'])

# Customers with product are more interested lead
plt.figure()
sns.countplot(df['Credit_Product'], hue=df['Is_Lead'])

# left skewed distribution and outliers should be taken care of
plt.figure()
sns.histplot(df['Avg_Account_Balance'])
plt.figure()
sns.boxplot(df['Avg_Account_Balance'])

# leads almost equally distributed
plt.figure()
sns.countplot(df['Is_Active'], hue=df['Is_Lead'])

################## PART A (Data Cleaning)####################################

# Step 1
# Identfying the datatypes 
data_types = df.dtypes
cols = df.columns


# Step 2
# Dropping obviously unwanted columns
df = df.drop('ID', axis=1)


# Step 3
# Finding missing values in the data
miss_vals = df.isnull().sum(axis=0)


# Filling the missing values with 'na' (not available)
# Rationale: most of the customers with missing values are interested in Credit Card
#            (so filling with separate value instead of 'Yes', or 'No')
df['Credit_Product'] = df['Credit_Product'].fillna('Missing')


# Step 4
# Converting Vintage (months to years)
df['Vintage'] = df['Vintage'] / 12


# Step 5
# Changing Region code and Channel code to numerical variables to reduce features for modeling
df['Region_Code'] = df['Region_Code'].str.replace('RG', '').astype(int)
df['Channel_Code'] = df['Channel_Code'].str.replace('X', '').astype(int)


# Step 6
# Outlier handling as seen in basic analysis
# Capping the Avg Account balance at 94th quantile

   
bal_upper_limit = df['Avg_Account_Balance'].quantile(0.94)
df.loc[df['Avg_Account_Balance'] > bal_upper_limit, 'Avg_Account_Balance'] = bal_upper_limit


# Step 7
# Checking for inter correlation among the features
corr_matrix = df.corr()


################################### PART B (Feature Selection)####################################

# copying cleaned data into new DataFrame
features_set = df.copy()

# Encoding categorical columns
features_set = pd.get_dummies(features_set, drop_first=True)

# Separating train and test data from featues dataset
train_set = features_set.iloc[:len(train_data), :]
test_set = features_set.iloc[len(train_data):, :]

# Creating Independent and Dependent Variables from the training set
X = train_set.drop('Is_Lead', axis=1)
Y = train_set['Is_Lead']
X_cols = X.columns


# Scaling the data
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X = mms.fit_transform(X)


# Using Chi Square analysis with null hypothesis as Y does not depend on each feature of X
from sklearn.feature_selection import SelectKBest, chi2

kbest = SelectKBest(chi2, k='all')
kbest.fit_transform(X, Y)
pvalues = kbest.pvalues_
pcols = pd.DataFrame(pvalues).rename(columns={0: 'p_cols'})


# At 95% confidence interval if p < 0.05 we reject the null hypothesis
# if pvalues > 0.05 the features could be irrelevant
selected_cols = pcols[pcols['p_cols'] < 0.05]
best_cols = selected_cols.index.values.tolist()

# Using All the Columns
X_a = train_set.drop('Is_Lead', axis=1)
X_b = X_a.iloc[:, best_cols]
# All the features are rejecting the null hypothesis, so we can use all the columns



################################### PART C (Model Selection)####################################

# Splitting the data for model evaluation
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_a, Y, test_size=0.1, random_state=1234, stratify=Y)

# Scaling the data
mms2 = MinMaxScaler()
x_train = mms2.fit_transform(x_train)
x_test = mms2.transform(x_test)


######################## RANDOMFOREST #################################
# Applying Machine Learning Models
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(class_weight='balanced', n_estimators=100)

# fitting data
rfc.fit(x_train, y_train)
y_predict_rfc = rfc.predict(x_test)
y_proba_rfc = rfc.predict_proba(x_test)
rfc_score = rfc.score(x_test, y_test)

# Analysing the model with metrics
from sklearn.metrics import confusion_matrix, roc_auc_score

result_score = roc_auc_score(y_test, y_proba_rfc[:, 1])
cm_rfc = confusion_matrix(y_test, y_predict_rfc)
fi_rfc = rfc.feature_importances_


###################### CATBOOST ###########################################
from catboost import CatBoostClassifier

cb = CatBoostClassifier(scale_pos_weight=3)
cb.fit(x_train, y_train)
y_predict_cb = cb.predict(x_test)
y_proba_cb = cb.predict_proba(x_test)
cb_score = cb.score(x_test, y_test)


fi_cb = cb.feature_importances_
cm_cb = confusion_matrix(y_test, y_predict_cb)
result_score_cb = roc_auc_score(y_test, y_proba_cb[:, 1])


############################ LIGHTGBM ###################################
from lightgbm import LGBMClassifier

#lgb = LGBMClassifier(scale_pos_weight=9, 
#                     num_leaves=20, 
#                     objective='binary', 
#                     learning_rate=0.1)


lgb = LGBMClassifier(scale_pos_weight=9, objective='binary')
lgb.fit(x_train, y_train)
y_predict_lgb = lgb.predict(x_test)
y_proba_lgb = lgb.predict_proba(x_test)
score_lgb = lgb.score(x_test, y_test)


cm_lgb = confusion_matrix(y_test, y_predict_lgb)
result_score_lgb = roc_auc_score(y_test, y_proba_lgb[:, 1])
fi_lgb = lgb.feature_importances_



######################## XGBOOST ##########################################
from xgboost import XGBClassifier

xgb = XGBClassifier(scale_pos_weight=9,
                    learning_rate=0.09,  
                    colsample_bytree = 0.8,
                    subsample = 0.9,
                    objective='binary:logistic', 
                    n_estimators=150,
                    max_depth=5,
                    gamma=0)
xgb.fit(x_train, y_train)
y_predict_xgb = xgb.predict(x_test)
y_proba_xgb = xgb.predict_proba(x_test)
score_xgb = xgb.score(x_test, y_test)

cm_xgb = confusion_matrix(y_test, y_predict_xgb)
result_score_xgb = roc_auc_score(y_test, y_proba_xgb[:, 1])
fi_xgb = xgb.feature_importances_


# Model Ranking
# 1. LightGBM
# 2. XGBoost
# 3. CatBoost
# 4. RandomForest



#Peforming Hyperparameter tuning for improving the LightGBM Model
#creating params for hyper parameter tuning
lgb2 = LGBMClassifier()
lgb_params = {
                'num_leaves': [8,12,16,20],
                'colsample_bytree' : [0.65, 0.66],
                'subsample' : [0.7,0.75],
             }


from sklearn.model_selection import GridSearchCV

gsc = GridSearchCV(estimator=lgb2, param_grid=lgb_params, cv=4, scoring='roc_auc', return_train_score=True)

gsc_fit = gsc.fit(X_a, Y)

cv_results_lgb = pd.DataFrame.from_dict(gsc_fit.cv_results_)
# The results are better with model's default parameters




################################### PART D (TEST DATA REPORTING)####################################

#Reading test data IDs
sample_data = pd.read_csv('sample_submission.csv')

sample_data = sample_data.drop('Is_Lead', axis=1)

test_set = test_set.drop('Is_Lead', axis=1)

#Scaling test data
test_set = mms2.transform(test_set)

#Predicitng test data with LightGBM Model
test_predict_lgb = lgb.predict_proba(test_set)

test_results_lgb = pd.DataFrame(test_predict_lgb[:, 1]).rename(columns={0: 'Is_Lead'})

final_results_lgb = pd.concat([sample_data, test_results_lgb], axis=1)

#Exporting results into csv file
export = final_results_lgb.to_csv('sub_final.csv')









