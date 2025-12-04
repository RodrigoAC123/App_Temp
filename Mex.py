import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la temperatura  ''')
st.image("temp.jpg", caption="Predicción de la temperatura de tu zona.")

st.header('Datos de la zona')

def user_input_features():
  # Entrada
  País = st.number_input('Elije el país (México=0):', min_value=1, max_value=1, value = 1, step = 1)
  Ciudad = st.number_input('Elije tu ciudad (Acapulco:0  Acuña:1  Aguascalientes:2): ',  min_value=0, max_value=2, value = 1, step = 1)
  Año = st.number_input('Año a predecir', min_value=1823, max_value=2026, value = 2000, step = 1)
  Mes = st.number_input('Mes a predecir:', min_value=1, max_value=12, value = 4, step = 1)
  


  user_input_data = {'Country': País,
                     'City': Ciudad,
                     'Year': Año,
                     'Month': Mes,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

datos =  pd.read_csv('Mex.csv', encoding='latin-1')
X = datos.drop(columns='AverageTemperature')
y = datos['AverageTemperature']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613726)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['City'] + b1[1]*df['Country'] + b1[2]*df['Year'] + b1[3]*df['Month'] 

st.subheader('Cálculo de la temperatura')
st.write('La temperatura es ', prediccion)
