# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:09:04 2025

@author: rportatil115
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

# --- Modelo RNN/LSTM con PyTorch ---
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, rnn_type='lstm'):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Tipo de RNN no soportado")
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- Funciones auxiliares ---
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
    # Limpieza robusta
    real = real.dropna()
    real = real[~real.index.duplicated(keep='first')]

    df_hist = pd.DataFrame({
        'Fecha': real.index,
        'Precio': real.values.flatten()
    })

    ultima_fecha = real.index[-1]
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=dias_futuros)

    df_pred = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Precio': prediccion.flatten() if isinstance(prediccion, np.ndarray) else list(prediccion)
    })

    color_pred = 'green' if sesgo > 0 else 'red' if sesgo < 0 else 'blue'

    # Crear gráfico
    fig = go.Figure()

    # Línea histórica
    fig.add_trace(go.Scatter(
        x=df_hist['Fecha'],
        y=df_hist['Precio'],
        mode='lines',
        name='Histórico',
        line=dict(color='white', width=2)
    ))

    # Línea predicción
    fig.add_trace(go.Scatter(
        x=df_pred['Fecha'],
        y=df_pred['Precio'],
        mode='lines+markers',
        name=f'Predicción (sesgo: {sesgo:.2f})',
        line=dict(color=color_pred, dash='dash'),
        marker=dict(color=color_pred, size=6)
    ))

    # Anotaciones
    fig.add_trace(go.Scatter(
        x=[df_hist['Fecha'].iloc[-1]],
        y=[df_hist['Precio'].iloc[-1]],
        mode='markers+text',
        name='Último real',
        marker=dict(color='white', size=10, symbol='circle'),
        text=[f"{df_hist['Precio'].iloc[-1]:.2f}"],
        textposition="top center",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[df_pred['Fecha'].iloc[0]],
        y=[df_pred['Precio'].iloc[0]],
        mode='markers+text',
        name='Primera predicción',
        marker=dict(color=color_pred, size=10, symbol='circle'),
        text=[f"{df_pred['Precio'].iloc[0]:.2f}"],
        textposition="top center",
        showlegend=False
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
        template='plotly_dark',
        margin=dict(t=100),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5
        )
    )

    return fig

# --- Pipeline principal ---
def ejecutar_pipeline(ticker, inicio, fin, dias_pred, sesgo=0.0, modelo='lstm'):
    datos = obtener_datos(ticker, inicio, fin)
    if datos.empty:
        st.error("No se pudieron obtener datos para ese ticker.")
        return None, None, None

    serie_escalada, scaler, train, test, serie_original = preprocesar_datos(datos)
    look_back = 60

    if modelo == 'arima':
        modelo_arima = ARIMA(serie_escalada, order=(5, 1, 0))
        modelo_fit = modelo_arima.fit()
        pred = modelo_fit.forecast(steps=dias_pred)

    elif modelo in ['rnn', 'lstm']:
        X_train = np.array([train[i:i + look_back] for i in range(len(train) - look_back)])
        Y_train = np.array([train[i + look_back] for i in range(len(train) - look_back)])
        X_train = torch.FloatTensor(X_train).unsqueeze(2)
        Y_train = torch.FloatTensor(Y_train).unsqueeze(1)

        model = RNNModel(rnn_type=modelo)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()

        ultimos_datos = list(serie_escalada[-look_back:])
        pred = []
        model.eval()
        with torch.no_grad():
            for i in range(dias_pred):
                x_input = torch.FloatTensor(ultimos_datos[-look_back:]).view(1, look_back, 1)
                yhat = model(x_input).item()
                if i == 0:
                    yhat = ultimos_datos[-1]
                pred.append(yhat)
                ultimos_datos.append(yhat)

    else:
        st.error("Modelo no soportado.")
        return None, None, None

    pred_ajustada = ajustar_sesgo(np.array(pred), sesgo)
    pred_final = scaler.inverse_transform(pred_ajustada.reshape(-1, 1)).flatten()

    serie_real = pd.Series(serie_original.values.flatten(), index=serie_original.index)
    fig = graficar_prediccion_futura(serie_real, pred_final, dias_pred, ticker, modelo, sesgo)

    mse_test = mean_squared_error(test[:len(test)], train[-len(test):])

    return fig, mse_test, pred_final

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Predicción de Acciones", layout="wide")

st.sidebar.title("⚙️ Parámetros de Predicción")

with st.sidebar.form("parametros_prediccion"):
    ticker = st.text_input("Símbolo del ticker (ej: AAPL)", value="AAPL")
    inicio = st.date_input("Fecha de inicio", value=pd.to_datetime("2020-01-01"))
    fin = st.date_input("Fecha de fin", value=pd.to_datetime("2024-12-31"))
    dias = st.number_input("Días a predecir", min_value=1, max_value=365, value=30)
    sesgo = st.slider("Sesgo (-1 a 1, pesimista a optimista)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    modelo = st.selectbox("Modelo a utilizar", options=["arima", "rnn", "lstm"], index=2)
    submitted = st.form_submit_button("Predecir")

st.title("📈 Sistema de Predicción de Acciones con Sesgo e Interactividad")

if submitted:
    with st.spinner("Procesando..."):
        fig, mse_test, pred_final = ejecutar_pipeline(
            ticker.strip().upper(),
            inicio.strftime("%Y-%m-%d"),
            fin.strftime("%Y-%m-%d"),
            int(dias),
            sesgo=sesgo,
            modelo=modelo
        )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"[Evaluación] MSE en conjunto de prueba: {mse_test:.6f}")

        # Mostrar tabla con últimos 5 días de la predicción
        ultimos_5_valores = pred_final[-5:]
        fechas_futuras = pd.date_range(
            start=pd.to_datetime(fin) + pd.Timedelta(days=dias - 4),
            periods=5
        )
        df_ultimos = pd.DataFrame({
            "Fecha": fechas_futuras.strftime('%Y-%m-%d'),
            "Precio Predicho (USD)": np.round(ultimos_5_valores, 2)
        })

        st.write("### Últimos 5 días de la predicción")
        st.table(df_ultimos)
else:
    st.info("Por favor, ingresa los parámetros y haz clic en 'Predecir' en la barra lateral para mostrar resultados.")
