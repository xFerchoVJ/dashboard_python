import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

@st.cache_data
def load_data(filename):
  data = pd.read_csv(filename)
  return data

def clean_data(data):
  columns_to_drop = ["index", "Unnamed: 22"]
  data = data.drop(columns=columns_to_drop, errors='ignore')
  data = data.drop_duplicates()
  data = data.dropna()
  return data

def plot_sales_by_category(data):
  sales_by_category = data['Category'].value_counts()
  st.subheader('Ventas por categoría')
  st.bar_chart(sales_by_category)

def plot_sales_over_time(data):
  data['Date'] = pd.to_datetime(data['Date'])  # Convertir la columna de fecha a tipo datetime si no está en ese formato
  data = data.set_index('Date')  # Establecer la fecha como índice para facilitar el trazado

  sales_over_time = data.resample('M').size()  # Agrupar las ventas por mes (puedes ajustar la frecuencia)
  st.subheader('Tendencia de ventas a lo largo del tiempo')
  st.line_chart(sales_over_time)

def plot_sales_by_fulfilment_method(data):
  sales_by_fulfilment = data['Fulfilment'].value_counts()
  st.subheader('Distribución de ventas por método de fulfilment')

  fig, ax = plt.subplots()
  ax.pie(sales_by_fulfilment, labels=sales_by_fulfilment.index, autopct='%1.1f%%', startangle=90)
  ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  
  st.pyplot(fig)

def plot_quantity_vs_amount(data):
  st.subheader('Relación entre cantidad de productos vendidos y monto de las ventas')
  fig, ax = plt.subplots()
  ax.scatter(data['Qty'], data['Amount'], alpha=0.5)
  ax.set_xlabel('Cantidad')
  ax.set_ylabel('Monto de ventas')
  st.pyplot(fig)

def plot_top_cities_and_states(data):
  st.subheader('Principales ciudades y estados de los clientes')

  # Obtener las 10 ciudades y estados principales
  top_cities = data['ship-city'].value_counts().head(10)
  top_states = data['ship-state'].value_counts().head(10)

  # Crear el gráfico de barras para las ciudades
  plt.figure(figsize=(10, 6))
  plt.bar(top_cities.index, top_cities.values, color='skyblue')
  plt.xticks(rotation=45, ha='right')
  plt.xlabel('Ciudades')
  plt.ylabel('Cantidad de Clientes')
  plt.title('Top 10 Ciudades de Clientes')
  st.pyplot(plt)

  # Crear el gráfico de barras para los estados
  plt.figure(figsize=(10, 6))
  plt.bar(top_states.index, top_states.values, color='lightgreen')
  plt.xticks(rotation=45, ha='right')
  plt.xlabel('Estados')
  plt.ylabel('Cantidad de Clientes')
  plt.title('Top 10 Estados de Clientes')
  st.pyplot(plt)

def decision_tree_regression(data):
  st.subheader('Modelo de Árbol de Decisión para predecir Qty')

  # Seleccionamos las características y la variable objetivo
  features = ['Amount', 'Category']  # Columnas que podrían influir en 'Qty'
  target = 'Qty'  # Variable a predecir

  # Manejo de columnas categóricas con 'get_dummies'
  data_encoded = pd.get_dummies(data[features + [target]])

  # Dividir los datos en conjunto de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(data_encoded.drop(target, axis=1), data_encoded[target], test_size=0.2, random_state=42)

  # Entrenar el modelo
  model = DecisionTreeRegressor(random_state=42)
  model.fit(X_train, y_train)

  # Predicciones
  predictions = model.predict(X_test)

  # Métricas de evaluación
  r2 = r2_score(y_test, predictions)
  mse = mean_squared_error(y_test, predictions)

  st.write(f'Coeficiente de determinación (R²): {r2:.4f}')
  st.write(f'Error cuadrático medio (MSE): {mse:.4f}')

  # Gráfico de valores reales vs predicciones
  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, predictions)
  plt.xlabel('Valores reales (Qty)')
  plt.ylabel('Predicciones (Qty)')
  plt.title('Valores reales vs Predicciones (Árbol de Decisión)')
  st.pyplot(plt)

def decision_tree_regression_single_feature(data):
  st.subheader('Modelo de Árbol de Decisión para predecir Qty usando Amount')

  # Seleccionamos la característica y la variable objetivo
  feature = 'Amount'  # Característica seleccionada
  target = 'Qty'  # Variable a predecir

  # Dividir los datos en conjunto de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(data[[feature]], data[target], test_size=0.2, random_state=42)

  # Entrenar el modelo
  model = DecisionTreeRegressor(random_state=42)
  model.fit(X_train, y_train)

  # Predicciones
  predictions = model.predict(X_test)

  # Métricas de evaluación
  r2 = r2_score(y_test, predictions)
  mse = mean_squared_error(y_test, predictions)

  st.write(f'Coeficiente de determinación (R²): {r2:.4f}')
  st.write(f'Error cuadrático medio (MSE): {mse:.4f}')

  # Gráfico de valores reales vs predicciones
  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, predictions)
  plt.xlabel('Valores reales (Qty)')
  plt.ylabel('Predicciones (Qty)')
  plt.title('Valores reales vs Predicciones (Árbol de Decisión - Amount)')
  st.pyplot(plt)

def decision_tree_regression_amount_distribution(data):
  st.subheader('Distribución de Amount y Predicciones de Qty')

  # Seleccionamos la característica y la variable objetivo
  feature = 'Amount'  # Característica seleccionada
  target = 'Qty'  # Variable a predecir

  # Dividir los datos en conjunto de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(data[[feature]], data[target], test_size=0.2, random_state=42)

  # Entrenar el modelo
  model = DecisionTreeRegressor(random_state=42)
  model.fit(X_train, y_train)

  # Predicciones
  predictions = model.predict(X_test)

  # Calcular métricas de evaluación
  r2 = r2_score(y_test, predictions)
  mse = mean_squared_error(y_test, predictions)

  # Gráfico de distribución de valores reales y predicciones
  plt.figure(figsize=(10, 6))
  plt.scatter(X_test, y_test, label='Valores reales', alpha=0.5)
  plt.scatter(X_test, predictions, color='red', label='Predicciones', alpha=0.5)
  plt.xlabel('Amount')
  plt.ylabel('Qty')
  plt.title('Distribución de Amount y Predicciones de Qty')
  plt.legend()

  # Mostrar métricas en el gráfico como anotaciones
  plt.text(0.1, 0.9, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
  plt.text(0.1, 0.8, f'MSE = {mse:.4f}', transform=plt.gca().transAxes)
  
  st.pyplot(plt)
  
def main():
  st.title('Dashboard - Melanie')
  uploaded_file = st.file_uploader('Cargar archivo CSV', type=['csv'])

  if uploaded_file is None:
    return st.subheader('Ingresa el CSV para poder visualizar los datos')

  data_load_State = st.text('Cargando datos...')
  data = load_data(uploaded_file)
  data_load_State.text('Datos cargados correctamente!')

  if st.checkbox('Mostrar Conjunto de datos original'):
    st.subheader('Conjunto de datos original')
    st.dataframe(data)

  cleaned_data = clean_data(data)

  if st.checkbox('Mostrar conjunto de datos limpios'):
    st.subheader('Conjunto de datos limpio')
    st.dataframe(cleaned_data)

  if st.checkbox('Dashboard Estadistico'):
    plot_sales_by_category(cleaned_data)
    plot_sales_over_time(cleaned_data)
    plot_sales_by_fulfilment_method(data)
    plot_quantity_vs_amount(cleaned_data)
    plot_top_cities_and_states(cleaned_data)

  if st.checkbox('Dashboard de Mineria'):
    st.subheader('Mineria de Datos')
    decision_tree_regression(cleaned_data)
    decision_tree_regression_single_feature(cleaned_data)
    decision_tree_regression_amount_distribution(cleaned_data)

if __name__ == "__main__":
  main()