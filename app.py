import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models, layers, utils, optimizers
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title('Boston Housing Price Calculator')
tdata = np.array([])
user_input = st.text_input('Enter the per capita crime rate by town',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the proportion of residential land (Over 25,000 sq.ft.)',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the proportion of non-retail business acres / Town)',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the Charles River dummy variable(1 if tract bounds river, 0 otherwise)',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the nitric oxides concentration (Parts Per 10 Million)',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the average number of rooms per dwelling',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the proportion of owner-occupied units built prior to 1940',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the weighted distances to five Boston employment centres',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the index of accessibility to radial highways',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the full-value property-tax rate per $10,000',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the pupil-teacher ratio by town',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',0)
tdata = np.append(tdata,user_input)
user_input = st.text_input('Enter the percentage of lower status of the population',0)
tdata = np.append(tdata,user_input)


#Model Data
(train_data,train_target),(test_data, test_target)=boston_housing.load_data()


#Feeding Data into model

model =load_model('House_Price_Predictor.h5')


y_predicted = model.predict([tdata.astype('float32').reshape((1,13)),])

if st.button("Calculate"):
    st.write(y_predicted[0,0])