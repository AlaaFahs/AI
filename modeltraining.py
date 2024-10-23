#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:02:43 2024

@author: user
"""


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from views.datasets import Dataset
from views.utilities import plots

class Modtrain:

    class variables:
        pass
    
    def __init__(self, enabled = None):
        
        self.enabled = enabled
        
    def view(self, model1):
        
        fruits = Dataset(enabled = False).view(Dataset(enabled = False).variables())

        y = fruits.iloc[:,0]
        X = fruits.iloc[:,3:8]

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
        
        add_selectbox = st.sidebar.selectbox(
            'What classifier you want to use?',
            ('k-NN', 'SVC', 'RandomForest'))
        
        if add_selectbox == "k-NN":
            
            st.sidebar.subheader("Model Hyperparameters")
            K = st.sidebar.number_input("N neighbors", 1, 50, step = 1, key = 'k') 
            if self.enabled == "Train":

                if st.sidebar.button("Train", key = 'Train'):
                    st.subheader("k-NN Results")       
                    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = K))
                    model.fit(X_train,y_train)
                    st.header("Acurracy")
                    st.write("On the train set : ", model.score(X_train,y_train))
                    st.write("On the test set : ", model.score(X_test,y_test))
                    #plots().confusion_matrix(model, X_test, y_test)
                    plots().fruitsplot(X_train["mass"],X_train["width"],X_train["height"],  y_train)
            else:
                if st.sidebar.button("Test", key = 'Test'):
                    
                    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = K))
                    classifier = model.fit(X_train,y_train)
                    prediction = classifier.predict(X_test)
                    st.write(prediction)                
            
        if add_selectbox == "SVC":
            
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
            kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key = 'kernel')
            gamma = st.sidebar.radio("Gamma (kernel coefficient)", ("scale","auto"), key = 'gamma')
            
            if self.enabled == "Train":
                if st.sidebar.button("Train", key = 'Train'):
                    st.subheader("SVM Results")
                    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C = C, gamma = gamma))
                    classifier = model.fit(X_train,y_train)        
                    st.header("Acurracy")
                    st.write("On the train set : ", model.score(X_train,y_train))
                    st.write("On the test set : ", model.score(X_test,y_test))
                    
                    #plots().confusion_matrix(model, X_test, y_test)
                    plots().fruitsplot(X_train["mass"],X_train["width"],X_train["height"],  y_train)

            else:
                if st.sidebar.button("Test", key = 'Test'):
                    st.subheader("SVM Results")
                    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C = C, gamma = gamma))
                    classifier = model.fit(X_train,y_train)        
                    prediction = classifier.predict(X_test)
                    st.write(prediction)
            
        if add_selectbox == "RandomForest":
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimator')            
            max_depth = st.sidebar.number_input("The maximum depth of trees", 1, 20, step = 1, key = 'max_depth')            
            #boostrap = st.sidebar.radio("Boostrap samples when building the trees", ('True','False'), key = 'boostrap')            
            if self.enabled == "Train":

                if st.sidebar.button("Train", key = 'Train'):
                    st.subheader("RandomForest")                   
                    model = make_pipeline(StandardScaler(), RandomForestClassifier(
                        n_estimators= n_estimators, max_depth= max_depth, bootstrap=True, n_jobs= 1))
                    model.fit(X_train,y_train)
                    st.header("Acurracy")
                    st.write("On the train set : ", model.score(X_train,y_train))
                    st.write("On the test set : ", model.score(X_test,y_test))
                    #plots().confusion_matrix(model, X_test, y_test)
                    plots().fruitsplot(X_train["mass"],X_train["width"],X_train["height"],  y_train)

            else:
                if st.sidebar.button("Test", key = 'Test'):
                
                    model = make_pipeline(StandardScaler(), RandomForestClassifier(
                        n_estimators= n_estimators, max_depth= max_depth, bootstrap=True, n_jobs= 1))
                    classifier = model.fit(X_train,y_train)       
                    prediction = classifier.predict(X_test)
                    st.write(prediction)        
        
        