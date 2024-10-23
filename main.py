# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from views.datasets import Dataset
from modeltraining import Modtrain
from about import About



class Model:
    
    option_1 = "Datasets"
    option_2 = "Model Training"
    option_3 = "Model Testing"
    option_4 = "About"    
    menuTitle = "Menu"
    

def global_view(model):
 
    
    with st.sidebar:
        menu_item = option_menu(model.menuTitle, [model.option_1,
                    model.option_2, model.option_3, model.option_4] )
    
    
    if menu_item == model.option_1 :
        
        Dataset(enabled = True).view(Dataset(enabled = True).variables())
        
    if menu_item == model.option_2 :
        
        Modtrain(enabled = "Train").view(Modtrain(enabled = "Train").variables())

    if menu_item == model.option_3 :
        
        Modtrain(enabled = "Test").view(Modtrain(enabled = "Test").variables())

    if menu_item == model.option_4 :
        
       About().view(About().variables())

        
    with st.sidebar:
        
        st.markdown("---")
        st.text("User: Alaa FAHS")
        st.text("Version 0.0.1")
        st.markdown("---")


global_view(Model())
    
    
    
    
    
    
    
    
    
    
    