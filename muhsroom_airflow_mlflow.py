#!/usr/bin/env python
# coding: utf-8

# In[58]:


import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime


# In[53]:


def prepro():
    mushroom = pd.read_csv('mushrooms.csv')
    mushroom.isna().sum()
    mushroom = mushroom[['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']]  # exclude 'class'
    cat_variables = [x for x in mushroom.columns]
    df = pd.get_dummies(data = mushroom,prefix = cat_variables,columns = cat_variables)
    df.rename(columns = {'class_e':'class'}, inplace = True)
    df = df.drop(['class_p'], axis=1)
    df.to_csv("preprocess_data.csv")
    


# In[54]:


prepro()


# In[ ]:





# In[55]:


def training_eval():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("rf_mushroom_classifier")
    df=pd.read_csv("preprocess_data.csv")
    X = df.drop('class',axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 55)
    n_estimators = [10, 50, 100]
    max_depths = [None, 5, 10]

    for n_estimator in n_estimators:
        for max_depth in max_depths:
            with mlflow.start_run(run_name="rf_mushroom_example",nested=True):
                desc = "Random Forest classification for muhsroom dataset"
                mlflow.set_tag("description", desc)
                # Create and fit model
                rf = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth)
                rf.fit(X_train, y_train)

                # Evaluate model and log metrics
                y_pred = rf.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mlflow.log_metric("mse", mse)

                # Log hyperparameters and trained model
                mlflow.log_params(rf.get_params())
                mlflow.sklearn.log_model(rf, "model")
                mlflow.sklearn.autolog()


# In[56]:


def model_dep():
    experiment_id = mlflow.get_experiment_by_name("rf_diabetes_hyperparameter_search").experiment_id
    all_runs = mlflow.search_runs(experiment_ids=experiment_id, order_by=["metrics.mse"])
    best_run_id=all_runs.run_id[0]
    best_run_id
    mlflow.register_model(f"runs:/{best_run_id}/model", "mushroom_rf_model")


# In[57]:




# Define DAG arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 5, 5),
    'retries': 1
}

# Instantiate DAG
dag = DAG('model_pipeline', default_args=default_args)

# Define tasks
preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=prepro,
    dag=dag
)

train_and_evaluate = PythonOperator(
    task_id='train_and_evaluate',
    python_callable=training_eval,
    dag=dag
)

deploy_model = PythonOperator(
    task_id='deploy_model',
    python_callable=model_dep,
    dag=dag
)

# Set task dependencies
preprocess_data >> train_and_evaluate >> deploy_model


# In[ ]:





# In[ ]:




