# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:04:07 2025

@author: rportatil115
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def obtener_datos(ticker, inicio, fin):
    datos = yf.download(ticker, start=inicio, end=fin)
    return datos

def preprocesar_datos(datos, columna='Close', proporcion_entrenamiento=0.8):
    serie = datos[columna].dropna()
    scaler = MinMaxScaler()
    serie_escalada = scaler.fit_transform(serie.values.reshape(-1, 1)).flatten()
    tamaño_entrenamiento = int(len(serie_escalada) * proporcion_entrenamiento)
    train = serie_escalada[:tamaño_entrenamiento]
    test = serie_escalada[tamaño_entrenamiento:]
    return serie_escalada, scaler, train, test, serie

def ajustar_sesgo(prediccion, sesgo=0.0):
    if not -1 <= sesgo <= 1:
        raise ValueError("El sesgo debe estar entre -1 y 1.")
    tendencia = np.mean(prediccion[-5:] - np.mean(prediccion[:5])) / len(prediccion)
    ajuste = sesgo * np.std(prediccion) * (1 + abs(tendencia)) * np.linspace(0, 1, len(prediccion))
    return prediccion + ajuste

def graficar_prediccion_futura(real, prediccion, dias_futuros, ticker, modelo, sesgo):
    if not isinstance(real, pd.Series):
        raise ValueError("La serie real debe ser un pd.Series con índice datetime")

    if len(prediccion) != dias_futuros:
        raise ValueError("La predicción y los días futuros no coinciden")

    ultima_fecha = real.index[-1]
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=dias_futuros)

    color_pred = 'green' if sesgo > 0 else 'red' if sesgo < 0 else 'blue'

    fig = go.Figure()

    # Datos reales
    fig.add_trace(go.Scatter(
        x=real.index,
        y=real.values.flatten(),
        mode='lines',
        name='Histórico',
        line=dict(color='black')
    ))

    # Predicción futura
    fig.add_trace(go.Scatter(
        x=fechas_futuras,
        y=prediccion.flatten() if isinstance(prediccion, np.ndarray) else prediccion,
        mode='lines+markers',
        name=f'Predicción (sesgo: {sesgo:.2f})',
        line=dict(color=color_pred, dash='dash')
    ))

    fig.update_layout(
        title=dict(
            text=f'Predicción de {ticker.upper()} con {modelo.upper()}<br><sup>Sesgo aplicado: {sesgo:.2f} {"(Optimista)" if sesgo > 0 else "(Pesimista)" if sesgo < 0 else "(Neutral)"}</sup>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        template='plotly_white',
        margin=dict(t=100),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5
        )
    )

    archivo = f"prediccion_{ticker}_{modelo}.html"
    fig.write_html(archivo, auto_open=True)


def ejecutar_pipeline(ticker, inicio, fin, dias_pred, sesgo=0.0, modelo='lstm'):
    datos = obtener_datos(ticker, inicio, fin)
    if datos.empty:
        raise ValueError("No se pudieron obtener datos para ese ticker.")

    serie_escalada, scaler, train, test, serie_original = preprocesar_datos(datos)
    look_back = 60

    if modelo == 'arima':
        modelo_arima = ARIMA(train, order=(5, 1, 0))
        modelo_fit = modelo_arima.fit()
        pred = modelo_fit.forecast(steps=dias_pred)

    elif modelo in ['rnn', 'lstm']:
        X_train = np.array([train[i:i + look_back] for i in range(len(train) - look_back)])
        Y_train = np.array([train[i + look_back] for i in range(len(train) - look_back)])

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = Sequential()
        if modelo == 'rnn':
            model.add(SimpleRNN(50, input_shape=(look_back, 1)))
        else:
            model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

        ultimos_datos = train[-look_back:].tolist()
        pred = []

        for _ in range(dias_pred):
            x_input = np.array(ultimos_datos[-look_back:]).reshape(1, look_back, 1)
            yhat = model.predict(x_input, verbose=0)[0][0]
            pred.append(yhat)
            ultimos_datos.append(yhat)

    else:
        raise ValueError("Modelo no soportado.")

    pred_ajustada = ajustar_sesgo(np.array(pred), sesgo)
    pred_final = scaler.inverse_transform(pred_ajustada.reshape(-1, 1)).flatten()

    serie_real = pd.Series(serie_original.values.flatten(), index=serie_original.index)
    graficar_prediccion_futura(serie_real, pred_final, dias_pred, ticker, modelo, sesgo)

    mse_test = mean_squared_error(test[:len(test)], train[-len(test):])
    print(f"[Evaluación] MSE en conjunto de prueba: {mse_test:.6f}")
    print(f"\nPredicción final (últimos 5 valores): {pred_final[-5:]}")
    return pred_final

def interfaz_usuario():
    print("Sistema de Predicción de Acciones con Sesgo e Interactividad")
    print("=" * 60)

    ticker = input("Símbolo del ticker (ej: AAPL): ").strip().upper()
    inicio = input("Fecha de inicio (YYYY-MM-DD): ").strip()
    fin = input("Fecha de fin (YYYY-MM-DD): ").strip()
    dias = int(input("Días a predecir: ").strip())

    while True:
        try:
            sesgo = float(input("Sesgo (-1 a 1, pesimista a optimista): ").strip())
            if -1 <= sesgo <= 1:
                break
            print("Sesgo inválido. Debe estar entre -1 y 1.")
        except ValueError:
            print("Ingrese un número válido.")

    print("\nModelos disponibles:")
    modelos = {'1': 'arima', '2': 'rnn', '3': 'lstm'}
    for k, v in modelos.items():
        print(f"{k}. {v.upper()}")

    opcion = input("Seleccione el modelo (1-3): ").strip()
    while opcion not in modelos:
        opcion = input("Opción inválida. Seleccione 1, 2 o 3: ").strip()

    print("\nProcesando...")
    ejecutar_pipeline(ticker, inicio, fin, dias, sesgo=sesgo, modelo=modelos[opcion])

if __name__ == "__main__":
    interfaz_usuario()
