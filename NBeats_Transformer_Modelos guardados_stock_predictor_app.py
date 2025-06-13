# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:41:57 2025

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
import os
import joblib
warnings.filterwarnings('ignore')

# --- Constantes de Configuración ---
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2024-06-10"
DEFAULT_PREDICT_DAYS = 30
DEFAULT_BIAS = 0.0
DEFAULT_NOISE_FACTOR = 0.01
DEFAULT_RNN_LOOK_BACK = 60
RNN_HIDDEN_SIZE = 100
RNN_NUM_LAYERS = 1
RNN_EPOCHS = 150
RNN_LEARNING_RATE = 0.001
ARIMA_ORDER = (5, 1, 0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelos ---
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=RNN_HIDDEN_SIZE, num_layers=RNN_NUM_LAYERS, rnn_type='lstm'):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(DEVICE)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(DEVICE)
        else:
            raise ValueError("Tipo de RNN no soportado. Debe ser 'lstm' o 'rnn'.")
        self.fc = nn.Linear(hidden_size, 1).to(DEVICE)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- TRANSFORMER ---
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=16, nhead=2, num_layers=1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        out = self.transformer_encoder(src)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- Guardado y carga de modelos y scaler ---
def guardar_modelo(model, scaler, ticker, modelo_nombre):
    os.makedirs("modelos_guardados", exist_ok=True)
    model_path = f"modelos_guardados/{ticker}_{modelo_nombre}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{modelo_nombre}_scaler.pkl"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    st.info(f"Modelo guardado en {model_path} y scaler en {scaler_path}")

def cargar_modelo(model_class, ticker, modelo_nombre, scaler_default, **model_kwargs):
    model_path = f"modelos_guardados/{ticker}_{modelo_nombre}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{modelo_nombre}_scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = model_class(**model_kwargs).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        scaler = joblib.load(scaler_path)
        st.success(f"Modelo y scaler cargados desde disco.")
        return model, scaler
    return None, scaler_default

# --- Funciones auxiliares ---
def obtener_datos(ticker: str, inicio: str, fin: str) -> pd.DataFrame:
    try:
        datos = yf.download(ticker, start=inicio, end=fin)
        return datos
    except Exception as e:
        st.error(f"Error al obtener datos para '{ticker}': {e}")
        return pd.DataFrame()

def preprocesar_datos(datos: pd.DataFrame, columna: str = 'Close', proporcion_entrenamiento: float = 0.8):
    if columna not in datos.columns:
        raise ValueError(f"La columna '{columna}' no se encuentra en los datos proporcionados.")
    serie = datos[columna].dropna()
    if serie.empty:
        raise ValueError("La serie de datos está vacía después de eliminar NaNs.")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    serie_escalada = scaler.fit_transform(serie.values.reshape(-1, 1)).flatten()
    tamaño_entrenamiento = int(len(serie_escalada) * proporcion_entrenamiento)
    train_scaled = serie_escalada[:tamaño_entrenamiento]
    test_scaled = serie_escalada[tamaño_entrenamiento:]
    return serie_escalada, scaler, train_scaled, test_scaled, serie

def create_sequences(data: np.ndarray, look_back: int):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return torch.FloatTensor(np.array(X)).unsqueeze(2).to(DEVICE), \
           torch.FloatTensor(np.array(Y)).unsqueeze(1).to(DEVICE)

def ajustar_sesgo(prediccion: np.ndarray, sesgo: float, noise_factor: float = DEFAULT_NOISE_FACTOR) -> np.ndarray:
    if not -1 <= sesgo <= 1:
        raise ValueError("El sesgo debe estar entre -1 y 1.")
    if len(prediccion) < 2:
        tendencia = 0
    else:
        tendencia = (prediccion[-1] - prediccion[0]) / len(prediccion)
    ajuste_base = sesgo * np.std(prediccion) * (1 + abs(tendencia))
    ajuste_volatilidad = np.random.normal(0, noise_factor * np.std(prediccion), len(prediccion))
    ajuste_gradual = np.linspace(0, 1, len(prediccion)) * ajuste_base * np.sign(sesgo) if sesgo != 0 else 0
    return prediccion + ajuste_gradual + ajuste_volatilidad

def graficar_prediccion_futura(real: pd.Series, prediccion: np.ndarray, dias_futuros: int, ticker: str, modelo: str, sesgo: float) -> go.Figure:
    real = real.dropna()
    real = real[~real.index.duplicated(keep='first')]
    if real.empty:
        st.warning("No hay datos históricos para graficar.")
        return go.Figure()
    df_hist = pd.DataFrame({
        'Fecha': real.index,
        'Precio': real.values.flatten()
    })
    ultima_fecha_historica = real.index[-1]
    fechas_futuras = pd.bdate_range(start=ultima_fecha_historica + pd.Timedelta(days=1), periods=dias_futuros)
    if len(prediccion) > len(fechas_futuras):
        prediccion = prediccion[:len(fechas_futuras)]
    elif len(prediccion) < len(fechas_futuras):
        fechas_futuras = fechas_futuras[:len(prediccion)]
    df_pred = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Precio': prediccion.flatten()
    })
    color_pred = 'green' if sesgo > 0 else 'red' if sesgo < 0 else 'blue'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist['Fecha'],
        y=df_hist['Precio'],
        mode='lines',
        name='Histórico',
        line=dict(color='white', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_pred['Fecha'],
        y=df_pred['Precio'],
        mode='lines+markers',
        name=f'Predicción (sesgo: {sesgo:.2f})',
        line=dict(color=color_pred, dash='dash'),
        marker=dict(color=color_pred, size=6)
    ))
    if not df_hist.empty:
        fig.add_trace(go.Scatter(
            x=[df_hist['Fecha'].iloc[-1]],
            y=[df_hist['Precio'].iloc[-1]],
            mode='markers+text',
            name='Último real',
            marker=dict(color='white', size=10, symbol='circle'),
            text=[f"Último real: {df_hist['Precio'].iloc[-1]:.2f}"],
            textposition="top center",
            showlegend=False
        ))
    if not df_pred.empty:
        fig.add_trace(go.Scatter(
            x=[df_pred['Fecha'].iloc[0]],
            y=[df_pred['Precio'].iloc[0]],
            mode='markers+text',
            name='Primera predicción',
            marker=dict(color=color_pred, size=10, symbol='circle'),
            text=[f"Primera pred: {df_pred['Precio'].iloc[0]:.2f}"],
            textposition="bottom center",
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
def ejecutar_pipeline(ticker: str, inicio: str, fin: str, dias_pred: int,
                      sesgo: float = DEFAULT_BIAS, modelo: str = 'lstm',
                      look_back: int = DEFAULT_RNN_LOOK_BACK, noise_factor: float = DEFAULT_NOISE_FACTOR):
    datos = obtener_datos(ticker, inicio, fin)
    if datos.empty:
        return None, None, None, None, None
    try:
        serie_escalada, scaler, train_scaled, test_scaled, serie_original = preprocesar_datos(datos)
    except ValueError as e:
        st.error(f"Error en el preprocesamiento de datos: {e}")
        return None, None, None, None, None
    pred_scaled = []
    mse_test = None
    first_predicted_unscaled_debug = np.nan

    if modelo == 'arima':
        try:
            modelo_arima = ARIMA(serie_escalada, order=ARIMA_ORDER)
            modelo_fit = modelo_arima.fit()
            pred_scaled = modelo_fit.forecast(steps=dias_pred)
            forecast_test_scaled = modelo_fit.predict(start=len(train_scaled), end=len(serie_escalada)-1)
            mse_test = mean_squared_error(test_scaled, forecast_test_scaled)
        except Exception as e:
            st.error(f"Error al entrenar o predecir con ARIMA: {e}")
            return None, None, None, None, None

    elif modelo == 'transformer':
        # Intenta cargar modelo y scaler
        model, scaler = cargar_modelo(
            TransformerModel, ticker, 'transformer', scaler,
            input_size=1, d_model=16, nhead=2, num_layers=1
        )
        if model is None:
            X_train, Y_train = create_sequences(train_scaled, look_back)
            model = TransformerModel(input_size=1, d_model=16, nhead=2, num_layers=1).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=RNN_LEARNING_RATE)
            st.write(f"Entrenando modelo TRANSFORMER en {DEVICE}...")
            progress_bar = st.progress(0)
            model.train()
            for epoch in range(RNN_EPOCHS):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)
                loss.backward()
                optimizer.step()
                progress_bar.progress((epoch + 1) / RNN_EPOCHS)
            progress_bar.empty()
            st.success("Entrenamiento completado.")
            guardar_modelo(model, scaler, ticker, 'transformer')
        ultimos_datos_sequence = list(serie_escalada[-look_back:])
        model.eval()
        with torch.no_grad():
            for i in range(dias_pred):
                x_input = torch.FloatTensor(ultimos_datos_sequence[-look_back:]).view(1, look_back, 1).to(DEVICE)
                yhat = model(x_input).item()
                pred_scaled.append(yhat)
                ultimos_datos_sequence.append(yhat)
        mse_test = np.nan  # Si quieres, puedes calcularlo usando test

    # ... (haz lo mismo para otros modelos si quieres guardarlos/cargarlos)

    else:
        st.error("Modelo no soportado. Por favor, elige 'arima', 'rnn', 'lstm' o 'transformer'.")
        return None, None, None, None, None

    pred_ajustada_scaled = ajustar_sesgo(np.array(pred_scaled), sesgo, noise_factor)
    pred_final = scaler.inverse_transform(pred_ajustada_scaled.reshape(-1, 1)).flatten()
    serie_real = pd.Series(serie_original.values.flatten(), index=serie_original.index)
    fig = graficar_prediccion_futura(serie_real, pred_final, dias_pred, ticker, modelo, sesgo)
    return pred_final, mse_test, serie_real, scaler, fig

# --- Interfaz de usuario ---
st.title("Predicción de Series Temporales con Modelos Avanzados")
st.sidebar.header("Configuración")

ticker = st.sidebar.text_input("Ticker", DEFAULT_TICKER)
inicio = st.sidebar.date_input("Fecha de inicio", pd.to_datetime(DEFAULT_START_DATE))
fin = st.sidebar.date_input("Fecha de fin", pd.to_datetime(DEFAULT_END_DATE))
dias_pred = st.sidebar.slider("Días a predecir", 7, 90, DEFAULT_PREDICT_DAYS)
sesgo = st.sidebar.slider("Sesgo de mercado", -1.0, 1.0, DEFAULT_BIAS, 0.05)
modelo = st.sidebar.selectbox(
    "Modelo a utilizar",
    ('arima', 'rnn', 'lstm', 'transformer')
)
look_back = st.sidebar.slider("Ventana de look-back", 10, 120, DEFAULT_RNN_LOOK_BACK)
noise_factor = st.sidebar.slider("Ruido (simulación volatilidad)", 0.0, 0.1, DEFAULT_NOISE_FACTOR, 0.01)

if st.button("Predecir"):
    with st.spinner("Ejecutando pipeline..."):
        pred_final, mse_test, serie_real, scaler, fig = ejecutar_pipeline(
            ticker, str(inicio), str(fin), dias_pred, sesgo, modelo, look_back, noise_factor
        )
        if pred_final is not None:
            st.plotly_chart(fig, use_container_width=True)
            if mse_test is not None and not np.isnan(mse_test):
                st.info(f"MSE en test: {mse_test:.6f}")
